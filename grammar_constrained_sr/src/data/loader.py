"""
Data loading and preprocessing module for financial time series.

This module provides functionality to load and preprocess OHLCV (Open, High, Low, Close, Volume)
data for multiple assets, preparing it for feature computation and symbolic regression.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class AssetData:
    """Container for a single asset's OHLCV data."""
    symbol: str
    data: pd.DataFrame
    start_date: datetime
    end_date: datetime
    
    def __post_init__(self):
        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.index = pd.to_datetime(self.data.index)
        self.start_date = self.data.index[0]
        self.end_date = self.data.index[-1]


class DataLoader:
    """
    Main class for loading and preprocessing financial data.
    
    Handles loading from various sources (CSV, Parquet, databases) and
    performing essential preprocessing steps.
    """
    
    def __init__(self, 
                 data_dir: Optional[Union[str, Path]] = None,
                 date_col: str = 'date',
                 symbol_col: str = 'symbol'):
        """
        Initialize the DataLoader.
        
        Args:
            data_dir: Directory containing data files
            date_col: Name of the date column
            symbol_col: Name of the symbol/asset identifier column
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self.date_col = date_col
        self.symbol_col = symbol_col
        self.assets: Dict[str, AssetData] = {}
        
    def load_from_csv(self, 
                     file_path: Union[str, Path],
                     symbol: Optional[str] = None) -> AssetData:
        """
        Load data for a single asset from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            symbol: Asset symbol (if not in file)
            
        Returns:
            AssetData object containing the loaded data
        """
        df = pd.read_csv(file_path, parse_dates=[self.date_col])
        df = df.set_index(self.date_col).sort_index()
        
        if symbol is None:
            if self.symbol_col in df.columns:
                symbol = df[self.symbol_col].iloc[0]
                df = df.drop(columns=[self.symbol_col])
            else:
                symbol = Path(file_path).stem
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col.lower() not in [c.lower() for c in df.columns]]
        if missing_cols:
            logger.warning(f"Missing columns in {symbol}: {missing_cols}")
        
        # Standardize column names to lowercase
        df.columns = [col.lower() for col in df.columns]
        
        return AssetData(symbol=symbol, data=df)
    
    def load_from_parquet(self, 
                         file_path: Union[str, Path],
                         symbol: Optional[str] = None) -> AssetData:
        """
        Load data for a single asset from a Parquet file.
        
        Args:
            file_path: Path to the Parquet file
            symbol: Asset symbol (if not in file)
            
        Returns:
            AssetData object containing the loaded data
        """
        df = pd.read_parquet(file_path)
        df = df.set_index(self.date_col).sort_index()
        
        if symbol is None:
            if self.symbol_col in df.columns:
                symbol = df[self.symbol_col].iloc[0]
                df = df.drop(columns=[self.symbol_col])
            else:
                symbol = Path(file_path).stem
        
        # Standardize column names to lowercase
        df.columns = [col.lower() for col in df.columns]
        
        return AssetData(symbol=symbol, data=df)
    
    def load_directory(self, 
                       data_dir: Optional[Union[str, Path]] = None,
                       file_pattern: str = "*.csv",
                       file_type: str = "csv") -> Dict[str, AssetData]:
        """
        Load all data files from a directory.
        
        Args:
            data_dir: Directory to load from (defaults to self.data_dir)
            file_pattern: Glob pattern for files to load
            file_type: Type of files ('csv' or 'parquet')
            
        Returns:
            Dictionary mapping symbol to AssetData
        """
        load_dir = self.data_dir if data_dir is None else Path(data_dir)
        
        if not load_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {load_dir}")
        
        files = list(load_dir.glob(file_pattern))
        logger.info(f"Found {len(files)} files matching {file_pattern} in {load_dir}")
        
        for file_path in files:
            try:
                if file_type == "csv":
                    asset_data = self.load_from_csv(file_path)
                elif file_type == "parquet":
                    asset_data = self.load_from_parquet(file_path)
                else:
                    raise ValueError(f"Unsupported file type: {file_type}")
                
                self.assets[asset_data.symbol] = asset_data
                logger.info(f"Loaded {asset_data.symbol}: {asset_data.start_date} to {asset_data.end_date}")
                
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {str(e)}")
        
        return self.assets
    
    def preprocess_asset(self, 
                        asset_data: AssetData,
                        forward_return_horizon: int = 1,
                        min_periods: int = 252) -> pd.DataFrame:
        """
        Preprocess a single asset's data for alpha discovery.
        
        Args:
            asset_data: AssetData object to preprocess
            forward_return_horizon: Number of days for forward return calculation
            min_periods: Minimum number of periods required for calculations
            
        Returns:
            Preprocessed DataFrame with computed features
        """
        df = asset_data.data.copy()
        
        # Compute log returns
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Compute forward returns
        df['forward_return'] = df['log_return'].shift(-forward_return_horizon)
        
        # Compute simple returns
        df['simple_return'] = df['close'].pct_change()
        df['forward_simple_return'] = df['simple_return'].shift(-forward_return_horizon)
        
        # Compute typical price (for volume-weighted features)
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Compute dollar volume
        df['dollar_volume'] = df['volume'] * df['typical_price']
        
        # Drop rows with NaN values that can't be filled
        df = df.dropna(subset=['log_return', 'forward_return'])
        
        # Ensure we have enough data
        if len(df) < min_periods:
            logger.warning(f"Asset {asset_data.symbol} has only {len(df)} periods, less than {min_periods}")
        
        return df
    
    def preprocess_all(self, 
                      forward_return_horizon: int = 1,
                      min_periods: int = 252) -> Dict[str, pd.DataFrame]:
        """
        Preprocess all loaded assets.
        
        Args:
            forward_return_horizon: Number of days for forward return calculation
            min_periods: Minimum number of periods required for calculations
            
        Returns:
            Dictionary mapping symbol to preprocessed DataFrame
        """
        preprocessed = {}
        
        for symbol, asset_data in self.assets.items():
            try:
                preprocessed[symbol] = self.preprocess_asset(
                    asset_data, 
                    forward_return_horizon=forward_return_horizon,
                    min_periods=min_periods
                )
            except Exception as e:
                logger.error(f"Failed to preprocess {symbol}: {str(e)}")
        
        return preprocessed
    
    def get_common_dates(self, 
                        preprocessed: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
        """
        Get the intersection of all available dates across assets.
        
        Args:
            preprocessed: Dictionary of preprocessed DataFrames
            
        Returns:
            DatetimeIndex of common dates
        """
        all_dates = [df.index for df in preprocessed.values()]
        common_dates = all_dates[0]
        
        for dates in all_dates[1:]:
            common_dates = common_dates.intersection(dates)
        
        return common_dates
    
    def align_assets(self, 
                    preprocessed: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Align all assets to a common date index.
        
        Args:
            preprocessed: Dictionary of preprocessed DataFrames
            
        Returns:
            MultiIndex DataFrame with (date, symbol) index
        """
        common_dates = self.get_common_dates(preprocessed)
        
        aligned_data = []
        for symbol, df in preprocessed.items():
            # Reindex to common dates
            df_aligned = df.reindex(common_dates)
            df_aligned['symbol'] = symbol
            aligned_data.append(df_aligned)
        
        # Combine all data
        combined = pd.concat(aligned_data, axis=0)
        combined = combined.set_index('symbol', append=True)
        combined = combined.reorder_levels(['symbol', 'date'])
        
        return combined


def load_sample_data(num_assets: int = 10, 
                     num_days: int = 1000,
                     start_date: str = "2020-01-01") -> Dict[str, pd.DataFrame]:
    """
    Generate sample synthetic data for testing.
    
    Args:
        num_assets: Number of synthetic assets to generate
        num_days: Number of trading days
        start_date: Start date for the synthetic data
        
    Returns:
        Dictionary mapping symbol to DataFrame with OHLCV data
    """
    from .synthetic import SyntheticDataGenerator
    
    generator = SyntheticDataGenerator(
        num_assets=num_assets,
        num_days=num_days,
        start_date=start_date
    )
    
    return generator.generate()


def load_from_csv(file_path: Union[str, Path], 
                  date_col: str = 'date',
                  symbol_col: str = 'symbol') -> Dict[str, AssetData]:
    """
    Convenience function to load data from a single CSV file.
    
    Args:
        file_path: Path to the CSV file
        date_col: Name of the date column
        symbol_col: Name of the symbol column
        
    Returns:
        Dictionary with single AssetData entry
    """
    loader = DataLoader(date_col=date_col, symbol_col=symbol_col)
    asset_data = loader.load_from_csv(file_path)
    return {asset_data.symbol: asset_data}


def load_from_parquet(file_path: Union[str, Path], 
                      date_col: str = 'date',
                      symbol_col: str = 'symbol') -> Dict[str, AssetData]:
    """
    Convenience function to load data from a single Parquet file.
    
    Args:
        file_path: Path to the Parquet file
        date_col: Name of the date column
        symbol_col: Name of the symbol column
        
    Returns:
        Dictionary with single AssetData entry
    """
    loader = DataLoader(date_col=date_col, symbol_col=symbol_col)
    asset_data = loader.load_from_parquet(file_path)
    return {asset_data.symbol: asset_data}
