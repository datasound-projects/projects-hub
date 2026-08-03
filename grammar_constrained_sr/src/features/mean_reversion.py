"""
Mean reversion-related primitives for stylized fact 10.

This module implements primitives related to:
- Mean reversion (Stylized Fact 10)
"""

import pandas as pd
import numpy as np
from typing import Optional
from dataclasses import dataclass
import logging

from .base import Primitive, PrimitiveConfig, RollingPrimitive

logger = logging.getLogger(__name__)


# ============================================================================
# Mean Reversion (Stylized Fact 10)
# ============================================================================

class HodrickPrescottCycle(RollingPrimitive):
    """
    Hodrick-Prescott cycle component.
    
    Extracts the deviation from a smooth trend, providing a pure mean-reversion
    target.
    
    Stylized Fact: Mean reversion (10)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None,
                 lambda_param: float = 1600.0):
        """
        Initialize HP cycle.
        
        Args:
            config: Primitive configuration
            lambda_param: Smoothing parameter (higher = smoother trend)
        """
        super().__init__(
            config=config or PrimitiveConfig(),
            name=f"hp_cycle_{lambda_param}",
            description=f"Hodrick-Prescott cycle component (lambda={lambda_param})"
        )
        self.lambda_param = lambda_param
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute HP cycle component.
        
        Args:
            data: Input data (returns or prices)
            window: Window size (not used for HP filter)
            
        Returns:
            DataFrame with HP cycle values
        """
        # HP filter is typically applied to the entire series, not rolling
        # For rolling HP, we need a different approach
        
        # For now, we'll use a simple approximation: deviation from rolling mean
        window = self._get_window(window) if window else 60
        
        # Compute rolling mean (trend)
        trend = self._rolling_mean(data, window)
        
        # Compute cycle (deviation from trend)
        cycle = data - trend
        
        return cycle


class ZeroCrossingRate(RollingPrimitive):
    """
    Zero-crossing rate.
    
    Measures how frequently demeaned returns change sign. A high rate
    indicates choppy, mean-reverting behavior.
    
    Stylized Fact: Mean reversion (10)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None):
        super().__init__(
            config=config or PrimitiveConfig(),
            name="zero_crossing_rate",
            description="Rate of sign changes in demeaned returns"
        )
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute zero-crossing rate.
        
        Args:
            data: Input data (returns)
            window: Window size
            
        Returns:
            DataFrame with zero-crossing rate values
        """
        window = self._get_window(window)
        min_periods = self._get_min_periods(None)
        
        def zero_crossing_rate(x):
            """Compute zero-crossing rate for a single window."""
            if len(x) < min_periods:
                return np.nan
            
            # Demean the signal
            x_demeaned = x - np.mean(x)
            
            # Count sign changes
            sign_changes = np.sum(np.diff(np.sign(x_demeaned)) != 0)
            
            # Normalize by window length
            return sign_changes / (len(x) - 1)
        
        result = data.rolling(
            window=window,
            min_periods=min_periods
        ).apply(zero_crossing_rate, raw=True)
        
        return result


class NormalizedDeviation(RollingPrimitive):
    """
    Normalized deviation from rolling mean.
    
    Measures how far current returns are from their local average, normalized
    by the rolling standard deviation (z-score).
    
    Stylized Fact: Mean reversion (10)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None):
        super().__init__(
            config=config or PrimitiveConfig(),
            name="normalized_deviation",
            description="Z-score of returns relative to rolling mean"
        )
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute normalized deviation.
        
        Args:
            data: Input data (returns)
            window: Window size
            
        Returns:
            DataFrame with normalized deviation values
        """
        window = self._get_window(window)
        
        # Compute rolling mean and standard deviation
        rolling_mean = self._rolling_mean(data, window)
        rolling_std = self._rolling_std(data, window)
        
        # Compute z-score
        result = (data - rolling_mean) / rolling_std.replace(0, np.nan)
        
        return result


class MeanReversionSignal(RollingPrimitive):
    """
    Combined mean reversion signal.
    
    Combines multiple mean reversion indicators into a single signal.
    
    Stylized Fact: Mean reversion (10)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None):
        super().__init__(
            config=config or PrimitiveConfig(),
            name="mean_reversion_signal",
            description="Combined mean reversion signal"
        )
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute mean reversion signal.
        
        Args:
            data: Input data (returns)
            window: Window size
            
        Returns:
            DataFrame with mean reversion signal values
        """
        window = self._get_window(window)
        
        # Compute normalized deviation (z-score)
        rolling_mean = self._rolling_mean(data, window)
        rolling_std = self._rolling_std(data, window)
        z_score = (data - rolling_mean) / rolling_std.replace(0, np.nan)
        
        # Compute zero-crossing rate
        def zero_crossing_rate(x):
            if len(x) < 2:
                return np.nan
            x_demeaned = x - np.mean(x)
            sign_changes = np.sum(np.diff(np.sign(x_demeaned)) != 0)
            return sign_changes / (len(x) - 1)
        
        zcr = data.rolling(
            window=window,
            min_periods=self._get_min_periods(None)
        ).apply(zero_crossing_rate, raw=True)
        
        # Combine signals (simple average)
        result = (z_score + zcr) / 2
        
        return result
