"""
Lead-lag effect primitives for stylized fact 12.

This module implements primitives related to:
- Lead-lag effects (Stylized Fact 12)
"""

import pandas as pd
import numpy as np
from typing import Optional
from dataclasses import dataclass
import logging

from .base import Primitive, PrimitiveConfig, RollingPrimitive

logger = logging.getLogger(__name__)


# ============================================================================
# Lead-Lag Effects (Stylized Fact 12)
# ============================================================================

class RollingBeta(RollingPrimitive):
    """
    Rolling beta to the equal-weighted market return.
    
    Measures the sensitivity of an asset to market movements. Large, liquid
    assets tend to react to new information faster than small, illiquid ones.
    
    Stylized Fact: Lead-lag effects (12)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None,
                 window: int = 60):
        super().__init__(
            config=config or PrimitiveConfig(window=window),
            name=f"rolling_beta_{window}",
            description=f"Rolling beta to market over {window} days"
        )
    
    def compute(self, 
                data: pd.DataFrame,
                market_returns: Optional[pd.DataFrame] = None,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute rolling beta.
        
        Args:
            data: Input data (asset returns)
            market_returns: Market returns (if not in data)
            window: Window size
            
        Returns:
            DataFrame with rolling beta values
        """
        window = self._get_window(window)
        
        if market_returns is None:
            if 'market_return' not in data.columns:
                raise ValueError("market_returns must be provided or 'market_return' must be in data")
            market_returns = data['market_return']
        
        # Compute rolling covariance
        cov = self._rolling_cov(data, market_returns, window)
        
        # Compute rolling market variance
        market_var = self._rolling_var(market_returns, window)
        
        # Compute beta
        result = cov / market_var.replace(0, np.nan)
        
        return result


class HouMoskowitzDelay(RollingPrimitive):
    """
    Hou-Moskowitz delay measure.
    
    Quantifies how much additional explanatory power lagged market returns
    have beyond contemporaneous market returns. High delay indicates an
    asset that reacts slowly to market-wide information.
    
    Stylized Fact: Lead-lag effects (12)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None,
                 window: int = 60,
                 lag: int = 1):
        super().__init__(
            config=config or PrimitiveConfig(window=window),
            name=f"hou_moskowitz_delay_{window}_lag_{lag}",
            description=f"Hou-Moskowitz delay measure over {window} days with lag {lag}"
        )
        self.lag = lag
    
    def compute(self, 
                data: pd.DataFrame,
                market_returns: Optional[pd.DataFrame] = None,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute Hou-Moskowitz delay measure.
        
        Args:
            data: Input data (asset returns)
            market_returns: Market returns (if not in data)
            window: Window size
            
        Returns:
            DataFrame with delay measure values
        """
        window = self._get_window(window)
        
        if market_returns is None:
            if 'market_return' not in data.columns:
                raise ValueError("market_returns must be provided or 'market_return' must be in data")
            market_returns = data['market_return']
        
        # Create lagged market returns
        lagged_market = market_returns.shift(self.lag)
        
        # Compute R-squared with contemporaneous market
        def rsquared_contemp(x, y):
            if len(x) < 2 or len(y) < 2:
                return np.nan
            
            # Fit regression: x ~ y
            A = np.vstack([y, np.ones(len(y))]).T
            coeffs, _, _, _ = np.linalg.lstsq(A, x, rcond=None)
            
            # Compute R-squared
            fitted = coeffs[0] * y + coeffs[1]
            ss_res = np.sum((x - fitted) ** 2)
            ss_tot = np.sum((x - np.mean(x)) ** 2)
            
            if ss_tot <= 0:
                return np.nan
            
            return 1 - ss_res / ss_tot
        
        # Compute R-squared with lagged market
        def rsquared_lagged(x, y):
            if len(x) < 2 or len(y) < 2:
                return np.nan
            
            # Align x and lagged y
            if len(x) != len(y):
                min_len = min(len(x), len(y))
                x = x[:min_len]
                y = y[:min_len]
            
            # Fit regression: x ~ y
            A = np.vstack([y, np.ones(len(y))]).T
            coeffs, _, _, _ = np.linalg.lstsq(A, x, rcond=None)
            
            # Compute R-squared
            fitted = coeffs[0] * y + coeffs[1]
            ss_res = np.sum((x - fitted) ** 2)
            ss_tot = np.sum((x - np.mean(x)) ** 2)
            
            if ss_tot <= 0:
                return np.nan
            
            return 1 - ss_res / ss_tot
        
        # Compute R-squared for contemporaneous and lagged
        r2_contemp = data.rolling(
            window=window,
            min_periods=self._get_min_periods(None)
        ).apply(lambda x: rsquared_contemp(x, market_returns.loc[x.index]), raw=False)
        
        r2_lagged = data.rolling(
            window=window,
            min_periods=self._get_min_periods(None)
        ).apply(lambda x: rsquared_lagged(x, lagged_market.loc[x.index]), raw=False)
        
        # Compute delay: additional explanatory power from lagged market
        result = r2_lagged - r2_contemp
        
        return result


class LeadLagEffect(RollingPrimitive):
    """
    Combined lead-lag effect measure.
    
    Combines beta and delay measures into a single lead-lag indicator.
    
    Stylized Fact: Lead-lag effects (12)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None):
        super().__init__(
            config=config or PrimitiveConfig(),
            name="lead_lag_effect",
            description="Combined lead-lag effect measure"
        )
    
    def compute(self, 
                data: pd.DataFrame,
                market_returns: Optional[pd.DataFrame] = None,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute lead-lag effect.
        
        Args:
            data: Input data (asset returns)
            market_returns: Market returns (if not in data)
            window: Window size
            
        Returns:
            DataFrame with lead-lag effect values
        """
        window = self._get_window(window) if window else 60
        
        if market_returns is None:
            if 'market_return' not in data.columns:
                raise ValueError("market_returns must be provided or 'market_return' must be in data")
            market_returns = data['market_return']
        
        # Compute beta
        cov = self._rolling_cov(data, market_returns, window)
        market_var = self._rolling_var(market_returns, window)
        beta = cov / market_var.replace(0, np.nan)
        
        # Compute delay (simplified version)
        lagged_market = market_returns.shift(1)
        
        def delay_measure(x, y):
            if len(x) < 2 or len(y) < 2:
                return np.nan
            
            # Simple correlation with lagged market
            return np.corrcoef(x, y)[0, 1]
        
        delay = data.rolling(
            window=window,
            min_periods=self._get_min_periods(None)
        ).apply(lambda x: delay_measure(x, lagged_market.loc[x.index]), raw=False)
        
        # Combine: assets with high beta and low delay are leaders
        # Assets with low beta and high delay are laggards
        result = beta - delay
        
        return result
