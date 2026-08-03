"""
Panel data construction module.

This module provides functionality to construct panel data from multiple assets,
ensuring proper alignment and handling of missing data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PanelConfig:
    """Configuration for panel construction."""
    forward_return_horizon: int = 1
    min_periods: int = 252
    max_missing_fraction: float = 0.1
    fill_method: Optional[str] = None  # 'ffill', 'bfill', or None
    

class PanelConstructor:
    """
    Constructs panel data from multiple assets for alpha discovery.
    
    Handles alignment, missing data, and preparation for feature computation.
    """
    
    def __init__(self, config: Optional[PanelConfig] = None):
        """
        Initialize the PanelConstructor.
        
        Args:
            config: Panel configuration
        """
        self.config = config or PanelConfig()
        
    def construct_from_dict(self, 
                           asset_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Construct panel from dictionary of asset DataFrames.
        
        Args:
            asset_data: Dictionary mapping symbol to DataFrame
            
        Returns:
            Panel DataFrame with MultiIndex (symbol, date)
        """
        # First, ensure all DataFrames have the same columns
        all_columns = set()
        for df in asset_data.values():
            all_columns.update(df.columns)
        
        # Find common columns (excluding symbol-specific ones)
        common_columns = ['open', 'high', 'low', 'close', 'volume', 
                         'log_return', 'forward_return', 'simple_return', 
                         'forward_simple_return', 'typical_price', 'dollar_volume']
        
        available_columns = [col for col in common_columns if col in all_columns]
        
        # Align all DataFrames to common columns
        aligned_data = {}
        for symbol, df in asset_data.items():
            # Select only available columns
            df_aligned = df[available_columns].copy()
            df_aligned['symbol'] = symbol
            aligned_data[symbol] = df_aligned
        
        # Concatenate all data
        panel = pd.concat(aligned_data.values(), axis=0)
        panel = panel.set_index('symbol', append=True)
        panel = panel.reorder_levels(['symbol', 'date'])
        
        # Sort by date and symbol
        panel = panel.sort_index()
        
        return panel
    
    def construct_from_loader(self, 
                             loader) -> pd.DataFrame:
        """
        Construct panel from a DataLoader instance.
        
        Args:
            loader: DataLoader with loaded assets
            
        Returns:
            Panel DataFrame
        """
        # Preprocess all assets
        preprocessed = loader.preprocess_all(
            forward_return_horizon=self.config.forward_return_horizon,
            min_periods=self.config.min_periods
        )
        
        # Construct panel
        panel = self.construct_from_dict(preprocessed)
        
        return panel
    
    def handle_missing_data(self, 
                           panel: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing data in the panel.
        
        Args:
            panel: Input panel DataFrame
            
        Returns:
            Panel with missing data handled
        """
        # Check for missing values
        total_cells = panel.size
        missing_cells = panel.isna().sum()
        missing_fraction = missing_cells / total_cells
        
        logger.info(f"Missing data: {missing_cells} cells ({missing_fraction:.2%})")
        
        if missing_fraction > self.config.max_missing_fraction:
            logger.warning(f"High missing data fraction: {missing_fraction:.2%} > {self.config.max_missing_fraction:.2%}")
        
        # Apply fill method if specified
        if self.config.fill_method:
            panel = panel.fillna(method=self.config.fill_method)
        
        # Drop rows with any remaining NaN values in critical columns
        critical_cols = ['close', 'volume', 'log_return', 'forward_return']
        available_critical = [col for col in critical_cols if col in panel.columns]
        
        if available_critical:
            panel = panel.dropna(subset=available_critical)
        
        return panel
    
    def validate_panel(self, 
                      panel: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate the constructed panel.
        
        Args:
            panel: Panel DataFrame to validate
            
        Returns:
            Tuple of (is_valid, message)
        """
        # Check index
        if not isinstance(panel.index, pd.MultiIndex):
            return False, "Panel must have MultiIndex"
        
        if panel.index.nlevels != 2:
            return False, f"Panel must have 2 index levels, got {panel.index.nlevels}"
        
        # Check index names
        if panel.index.names != ['symbol', 'date']:
            return False, f"Panel index names should be ['symbol', 'date'], got {panel.index.names}"
        
        # Check for empty panel
        if len(panel) == 0:
            return False, "Panel is empty"
        
        # Check for required columns
        required_cols = ['close', 'volume']
        missing_cols = [col for col in required_cols if col not in panel.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
        
        # Check for forward returns
        if 'forward_return' not in panel.columns:
            return False, "Panel must have forward_return column"
        
        # Check for NaN in critical columns
        critical_cols = ['close', 'volume', 'forward_return']
        available_critical = [col for col in critical_cols if col in panel.columns]
        
        for col in available_critical:
            nan_count = panel[col].isna().sum()
            if nan_count > 0:
                return False, f"Column '{col}' has {nan_count} NaN values"
        
        return True, "Panel is valid"
    
    def get_panel_stats(self, 
                       panel: pd.DataFrame) -> Dict[str, float]:
        """
        Get statistics about the panel.
        
        Args:
            panel: Panel DataFrame
            
        Returns:
            Dictionary of panel statistics
        """
        stats = {}
        
        # Basic stats
        stats['num_assets'] = panel.index.get_level_values('symbol').nunique()
        stats['num_dates'] = panel.index.get_level_values('date').nunique()
        stats['total_observations'] = len(panel)
        
        # Date range
        dates = panel.index.get_level_values('date')
        stats['start_date'] = dates.min()
        stats['end_date'] = dates.max()
        stats['date_range_days'] = (dates.max() - dates.min()).days
        
        # Forward return stats
        if 'forward_return' in panel.columns:
            forward_returns = panel['forward_return']
            stats['forward_return_mean'] = forward_returns.mean()
            stats['forward_return_std'] = forward_returns.std()
            stats['forward_return_min'] = forward_returns.min()
            stats['forward_return_max'] = forward_returns.max()
        
        # Volume stats
        if 'volume' in panel.columns:
            volumes = panel['volume']
            stats['volume_mean'] = volumes.mean()
            stats['volume_std'] = volumes.std()
        
        return stats


def create_panel(asset_data: Dict[str, pd.DataFrame],
                forward_return_horizon: int = 1,
                min_periods: int = 252) -> pd.DataFrame:
    """
    Convenience function to create a panel from asset data.
    
    Args:
        asset_data: Dictionary mapping symbol to DataFrame
        forward_return_horizon: Number of days for forward return
        min_periods: Minimum number of periods
        
    Returns:
        Panel DataFrame
    """
    constructor = PanelConstructor(
        PanelConfig(
            forward_return_horizon=forward_return_horizon,
            min_periods=min_periods
        )
    )
    
    panel = constructor.construct_from_dict(asset_data)
    panel = constructor.handle_missing_data(panel)
    
    is_valid, message = constructor.validate_panel(panel)
    if not is_valid:
        raise ValueError(f"Invalid panel: {message}")
    
    return panel
