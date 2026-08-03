"""
Base classes for feature primitives.

This module defines the base classes and interfaces for all stylized fact primitives.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class PrimitiveConfig:
    """Base configuration for a primitive."""
    window: int = 20
    min_periods: int = 10
    center: bool = False
    

@dataclass
class FeatureConfig:
    """Configuration for feature computation."""
    # Window sizes for rolling calculations
    window_sizes: List[int] = field(default_factory=lambda: [5, 10, 20, 60])
    
    # Whether to include raw features (not normalized)
    include_raw: bool = True
    
    # Whether to include cross-sectional rank normalized features
    include_rank: bool = True
    
    # Whether to include z-score normalized features
    include_zscore: bool = True
    
    # Minimum periods for rolling calculations
    min_periods: int = 10
    
    # Whether to use numba for acceleration
    use_numba: bool = True
    
    # Whether to include wavelet features (computationally expensive)
    include_wavelet: bool = True
    
    # Whether to include Hilbert transform features
    include_hilbert: bool = True
    
    # Whether to include spectral features
    include_spectral: bool = True


class Primitive(ABC):
    """
    Abstract base class for all stylized fact primitives.
    
    Each primitive should:
    1. Take a (T x N) panel of data as input
    2. Compute a feature that captures a specific stylized fact
    3. Return a (T x N) panel of feature values
    4. Handle NaN values appropriately
    5. Be numerically stable
    6. Produce bounded outputs where possible
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None,
                 name: Optional[str] = None,
                 description: Optional[str] = None):
        """
        Initialize the primitive.
        
        Args:
            config: Configuration for the primitive
            name: Name of the primitive
            description: Description of what the primitive measures
        """
        self.config = config or PrimitiveConfig()
        self.name = name or self.__class__.__name__
        self.description = description or ""
        
    @abstractmethod
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute the primitive for the given data.
        
        Args:
            data: Input data (T x N panel)
            window: Window size (overrides config.window if provided)
            
        Returns:
            DataFrame with computed feature values
        """
        pass
    
    def compute_for_windows(self, 
                          data: pd.DataFrame,
                          windows: List[int]) -> Dict[int, pd.DataFrame]:
        """
        Compute the primitive for multiple window sizes.
        
        Args:
            data: Input data
            windows: List of window sizes
            
        Returns:
            Dictionary mapping window size to feature DataFrame
        """
        results = {}
        for window in windows:
            try:
                result = self.compute(data, window=window)
                results[window] = result
            except Exception as e:
                logger.error(f"Failed to compute {self.name} for window {window}: {str(e)}")
        
        return results
    
    def _get_window(self, window: Optional[int]) -> int:
        """Get the window size to use."""
        return window if window is not None else self.config.window
    
    def _get_min_periods(self, min_periods: Optional[int]) -> int:
        """Get the minimum periods to use."""
        return min_periods if min_periods is not None else self.config.min_periods


class RollingPrimitive(Primitive):
    """
    Base class for primitives that use rolling window calculations.
    
    Provides common functionality for rolling computations.
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None,
                 name: Optional[str] = None,
                 description: Optional[str] = None):
        super().__init__(config, name, description)
    
    def _rolling_apply(self, 
                      data: pd.DataFrame,
                      func: callable,
                      window: int,
                      min_periods: Optional[int] = None,
                      center: bool = False) -> pd.DataFrame:
        """
        Apply a function in a rolling window.
        
        Args:
            data: Input data
            func: Function to apply
            window: Window size
            min_periods: Minimum periods
            center: Whether to center the window
            
        Returns:
            DataFrame with rolling results
        """
        min_periods = self._get_min_periods(min_periods)
        
        # Apply rolling function
        result = data.rolling(
            window=window,
            min_periods=min_periods,
            center=center
        ).apply(func, raw=True)
        
        return result
    
    def _rolling_mean(self, 
                     data: pd.DataFrame,
                     window: int,
                     min_periods: Optional[int] = None) -> pd.DataFrame:
        """Compute rolling mean."""
        min_periods = self._get_min_periods(min_periods)
        return data.rolling(window=window, min_periods=min_periods).mean()
    
    def _rolling_std(self, 
                    data: pd.DataFrame,
                    window: int,
                    min_periods: Optional[int] = None) -> pd.DataFrame:
        """Compute rolling standard deviation."""
        min_periods = self._get_min_periods(min_periods)
        return data.rolling(window=window, min_periods=min_periods).std()
    
    def _rolling_var(self, 
                    data: pd.DataFrame,
                    window: int,
                    min_periods: Optional[int] = None) -> pd.DataFrame:
        """Compute rolling variance."""
        min_periods = self._get_min_periods(min_periods)
        return data.rolling(window=window, min_periods=min_periods).var()
    
    def _rolling_corr(self, 
                     data1: pd.DataFrame,
                     data2: pd.DataFrame,
                     window: int,
                     min_periods: Optional[int] = None) -> pd.DataFrame:
        """Compute rolling correlation."""
        min_periods = self._get_min_periods(min_periods)
        return data1.rolling(window=window, min_periods=min_periods).corr(data2)
    
    def _rolling_cov(self, 
                    data1: pd.DataFrame,
                    data2: pd.DataFrame,
                    window: int,
                    min_periods: Optional[int] = None) -> pd.DataFrame:
        """Compute rolling covariance."""
        min_periods = self._get_min_periods(min_periods)
        return data1.rolling(window=window, min_periods=min_periods).cov(data2)
    
    def _rolling_autocorr(self, 
                         data: pd.DataFrame,
                         lag: int = 1,
                         window: int = 20,
                         min_periods: Optional[int] = None) -> pd.DataFrame:
        """Compute rolling autocorrelation at a given lag."""
        min_periods = self._get_min_periods(min_periods)
        
        def autocorr(x):
            if len(x) < lag + 1:
                return np.nan
            return np.corrcoef(x[:-lag], x[lag:])[0, 1]
        
        # Use expanding window for autocorrelation
        result = data.rolling(window=window, min_periods=min_periods).apply(
            lambda x: autocorr(x), raw=True
        )
        
        return result


class PanelPrimitive(Primitive):
    """
    Base class for primitives that operate on panel data.
    
    Handles the panel structure with MultiIndex (symbol, date).
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None,
                 name: Optional[str] = None,
                 description: Optional[str] = None):
        super().__init__(config, name, description)
    
    def _extract_column(self, 
                        panel: pd.DataFrame,
                        column: str) -> pd.DataFrame:
        """
        Extract a column from the panel and reshape for computation.
        
        Args:
            panel: Input panel DataFrame
            column: Column name to extract
            
        Returns:
            DataFrame with the column data
        """
        if column not in panel.columns:
            raise ValueError(f"Column '{column}' not found in panel")
        
        return panel[column].unstack(level='symbol')
    
    def _restore_panel_structure(self, 
                                  result: pd.DataFrame,
                                  panel: pd.DataFrame) -> pd.DataFrame:
        """
        Restore the panel structure to a computed result.
        
        Args:
            result: Computed result DataFrame
            panel: Original panel DataFrame
            
        Returns:
            Result with panel structure restored
        """
        # Stack the result to get back to (date, symbol) structure
        result_stacked = result.stack()
        result_stacked.name = self.name
        
        # Reindex to match original panel index
        result_stacked = result_stacked.reindex(panel.index, method='ffill')
        
        return result_stacked.to_frame()
    
    def compute(self, 
                panel: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute the primitive for panel data.
        
        Args:
            panel: Input panel DataFrame with MultiIndex (symbol, date)
            window: Window size
            
        Returns:
            Panel DataFrame with computed feature
        """
        # This should be implemented by subclasses
        pass
