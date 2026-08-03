"""
Momentum-related primitives for stylized fact 11.

This module implements primitives related to:
- Momentum (Stylized Fact 11)
"""

import pandas as pd
import numpy as np
from typing import Optional, List
from dataclasses import dataclass
import logging

from .base import Primitive, PrimitiveConfig, RollingPrimitive

logger = logging.getLogger(__name__)


# ============================================================================
# Momentum (Stylized Fact 11)
# ============================================================================

class RollingReturn(RollingPrimitive):
    """
    Rolling return over a specified window.
    
    Cumulative return over a rolling window.
    
    Stylized Fact: Momentum (11)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None,
                 window: int = 20):
        super().__init__(
            config=config or PrimitiveConfig(window=window),
            name=f"rolling_return_{window}",
            description=f"Rolling return over {window} days"
        )
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute rolling return.
        
        Args:
            data: Input data (returns)
            window: Window size
            
        Returns:
            DataFrame with rolling return values
        """
        window = self._get_window(window)
        
        # Compute cumulative return over window
        result = (1 + data).rolling(window=window).apply(
            lambda x: np.prod(x) - 1, raw=True
        )
        
        return result


class CumulativeReturn(RollingPrimitive):
    """
    Cumulative return over multiple windows.
    
    Computes cumulative returns over multiple standard windows (20, 60, 120, 252 days).
    
    Stylized Fact: Momentum (11)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None,
                 windows: List[int] = [20, 60, 120, 252]):
        super().__init__(
            config=config or PrimitiveConfig(),
            name="cumulative_return",
            description="Cumulative returns over multiple windows"
        )
        self.windows = windows
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute cumulative returns for all windows.
        
        Args:
            data: Input data (returns)
            window: Window size (not used, uses self.windows)
            
        Returns:
            DataFrame with cumulative return values for all windows
        """
        results = {}
        
        for w in self.windows:
            rolling_return = (1 + data).rolling(window=w).apply(
                lambda x: np.prod(x) - 1, raw=True
            )
            results[f"cumulative_return_{w}"] = rolling_return
        
        return pd.concat(results, axis=1)


class TrendStrength(RollingPrimitive):
    """
    Trend strength.
    
    Rolling Sharpe ratio, measuring the strength of the trend relative to
    its volatility.
    
    Stylized Fact: Momentum (11)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None,
                 window: int = 60):
        super().__init__(
            config=config or PrimitiveConfig(window=window),
            name=f"trend_strength_{window}",
            description=f"Trend strength (Sharpe ratio) over {window} days"
        )
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute trend strength (rolling Sharpe ratio).
        
        Args:
            data: Input data (returns)
            window: Window size
            
        Returns:
            DataFrame with trend strength values
        """
        window = self._get_window(window)
        
        # Compute rolling mean and standard deviation
        rolling_mean = self._rolling_mean(data, window)
        rolling_std = self._rolling_std(data, window)
        
        # Compute Sharpe ratio (annualized)
        # Assuming 252 trading days per year
        annualization_factor = np.sqrt(252 / window)
        
        result = rolling_mean / rolling_std.replace(0, np.nan) * annualization_factor
        
        return result


class TrendLinearity(RollingPrimitive):
    """
    Trend linearity.
    
    R-squared of a linear fit to cumulative returns, measuring how clean
    the trend is.
    
    Stylized Fact: Momentum (11)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None,
                 window: int = 60):
        super().__init__(
            config=config or PrimitiveConfig(window=window),
            name=f"trend_linearity_{window}",
            description=f"R-squared of linear fit to cumulative returns over {window} days"
        )
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute trend linearity.
        
        Args:
            data: Input data (returns)
            window: Window size
            
        Returns:
            DataFrame with trend linearity values
        """
        window = self._get_window(window)
        min_periods = self._get_min_periods(None)
        
        def trend_linearity(x):
            """Compute R-squared of linear fit to cumulative returns."""
            if len(x) < min_periods:
                return np.nan
            
            # Compute cumulative returns
            cum_returns = np.cumsum(x)
            
            # Create time indices
            t = np.arange(len(cum_returns))
            
            # Fit linear regression
            A = np.vstack([t, np.ones(len(t))]).T
            coeffs, _, _, _ = np.linalg.lstsq(A, cum_returns, rcond=None)
            
            # Compute fitted values
            fitted = coeffs[0] * t + coeffs[1]
            
            # Compute R-squared
            ss_res = np.sum((cum_returns - fitted) ** 2)
            ss_tot = np.sum((cum_returns - np.mean(cum_returns)) ** 2)
            
            if ss_tot <= 0:
                return np.nan
            
            return 1 - ss_res / ss_tot
        
        result = data.rolling(
            window=window,
            min_periods=min_periods
        ).apply(trend_linearity, raw=True)
        
        return result


class JegadeeshTitmanMomentum(RollingPrimitive):
    """
    Jegadeesh-Titman momentum measure.
    
    12-month cumulative return minus the most recent month, to avoid
    short-term reversal contamination.
    
    Stylized Fact: Momentum (11)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None,
                 long_window: int = 252,
                 short_window: int = 20):
        super().__init__(
            config=config or PrimitiveConfig(),
            name=f"jt_momentum_{long_window}_{short_window}",
            description=f"Jegadeesh-Titman momentum: {long_window}-day return minus {short_window}-day return"
        )
        self.long_window = long_window
        self.short_window = short_window
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute Jegadeesh-Titman momentum.
        
        Args:
            data: Input data (returns)
            window: Window size (not used)
            
        Returns:
            DataFrame with Jegadeesh-Titman momentum values
        """
        # Compute long-term cumulative return
        long_return = (1 + data).rolling(window=self.long_window).apply(
            lambda x: np.prod(x) - 1, raw=True
        )
        
        # Compute short-term cumulative return
        short_return = (1 + data).rolling(window=self.short_window).apply(
            lambda x: np.prod(x) - 1, raw=True
        )
        
        # Compute Jegadeesh-Titman momentum
        result = long_return - short_return
        
        return result
