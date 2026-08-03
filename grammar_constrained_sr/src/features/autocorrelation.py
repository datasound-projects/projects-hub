"""
Autocorrelation-related primitives for stylized facts 5-6.

This module implements primitives related to:
- Absence of linear autocorrelation (Stylized Fact 5)
- Slow decay of autocorrelation in absolute returns (Stylized Fact 6)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import logging

from .base import Primitive, PrimitiveConfig, RollingPrimitive

logger = logging.getLogger(__name__)


# ============================================================================
# Absence of Linear Autocorrelation (Stylized Fact 5)
# ============================================================================

class RollingAutocorrelation(RollingPrimitive):
    """
    Rolling first-order autocorrelation of returns.
    
    Returns themselves exhibit near-zero autocorrelation, consistent with
    weak-form market efficiency. However, small deviations from zero are
    exploitable and time-varying.
    
    Stylized Fact: Absence of linear autocorrelation (5)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None,
                 lag: int = 1):
        super().__init__(
            config=config or PrimitiveConfig(),
            name=f"rolling_autocorr_lag_{lag}",
            description=f"Rolling first-order autocorrelation of returns at lag {lag}"
        )
        self.lag = lag
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute rolling autocorrelation.
        
        Args:
            data: Input data (returns)
            window: Window size
            
        Returns:
            DataFrame with rolling autocorrelation values
        """
        window = self._get_window(window)
        min_periods = self._get_min_periods(None)
        
        def autocorr(x):
            """Compute autocorrelation at specified lag."""
            if len(x) < self.lag + 1:
                return np.nan
            return np.corrcoef(x[:-self.lag], x[self.lag:])[0, 1]
        
        result = data.rolling(
            window=window,
            min_periods=min_periods
        ).apply(autocorr, raw=True)
        
        return result


class VarianceRatio(RollingPrimitive):
    """
    Lo-MacKinlay variance ratio.
    
    Tests whether multi-period return variance scales linearly with the
    holding period (as it would under a random walk). Values above 1
    suggest momentum, below 1 suggest mean reversion.
    
    Stylized Fact: Absence of linear autocorrelation (5)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None,
                 k: int = 2):
        """
        Initialize variance ratio.
        
        Args:
            config: Primitive configuration
            k: Multiplier for holding period (e.g., k=2 for 2-period returns)
        """
        super().__init__(
            config=config or PrimitiveConfig(),
            name=f"variance_ratio_k_{k}",
            description=f"Lo-MacKinlay variance ratio for k={k}"
        )
        self.k = k
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute variance ratio.
        
        Args:
            data: Input data (returns)
            window: Window size
            
        Returns:
            DataFrame with variance ratio values
        """
        window = self._get_window(window)
        min_periods = self._get_min_periods(None)
        
        def variance_ratio(x):
            """Compute variance ratio for a single window."""
            if len(x) < window:
                return np.nan
            
            # Compute variance of k-period returns
            k_period_returns = x[::self.k]  # Every k-th observation
            if len(k_period_returns) < 2:
                return np.nan
            
            var_k = np.var(k_period_returns, ddof=1)
            
            # Compute variance of 1-period returns
            var_1 = np.var(x, ddof=1)
            
            # Variance ratio
            if var_1 <= 0:
                return np.nan
            
            return var_k / (self.k * var_1)
        
        result = data.rolling(
            window=window,
            min_periods=min_periods
        ).apply(variance_ratio, raw=True)
        
        return result


# ============================================================================
# Slow Decay of Autocorrelation in Absolute Returns (Stylized Fact 6)
# ============================================================================

class AutocorrelationDecay(RollingPrimitive):
    """
    Autocorrelation decay of absolute returns.
    
    Unlike returns themselves, absolute returns exhibit long-range dependence:
    their autocorrelation decays slowly, following a power law rather than an
    exponential.
    
    Stylized Fact: Slow decay of autocorrelation in absolute returns (6)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None,
                 max_lag: int = 10):
        super().__init__(
            config=config or PrimitiveConfig(),
            name=f"autocorr_decay_maxlag_{max_lag}",
            description=f"Autocorrelation decay of absolute returns (max lag={max_lag})"
        )
        self.max_lag = max_lag
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute autocorrelation decay.
        
        Args:
            data: Input data (returns)
            window: Window size
            
        Returns:
            DataFrame with autocorrelation decay values
        """
        window = self._get_window(window)
        min_periods = self._get_min_periods(None)
        
        def autocorr_decay(x):
            """Compute autocorrelation decay for a single window."""
            if len(x) < window:
                return np.nan
            
            abs_x = np.abs(x)
            
            # Compute autocorrelations at different lags
            acf_values = []
            for lag in range(1, min(self.max_lag + 1, len(abs_x))):
                if len(abs_x) < lag + 1:
                    break
                acf = np.corrcoef(abs_x[:-lag], abs_x[lag:])[0, 1]
                acf_values.append(acf)
            
            if len(acf_values) < 2:
                return np.nan
            
            # Fit a power law: acf(lag) = a * lag^(-b)
            # We want to estimate the decay rate b
            lags = np.arange(1, len(acf_values) + 1)
            log_acf = np.log(np.abs(acf_values) + 1e-10)
            log_lags = np.log(lags)
            
            # Linear regression in log-log space
            A = np.vstack([log_lags, np.ones(len(log_lags))]).T
            b, log_a = np.linalg.lstsq(A, log_acf, rcond=None)[0]
            
            return -b  # Return the decay exponent
        
        result = data.rolling(
            window=window,
            min_periods=min_periods
        ).apply(autocorr_decay, raw=True)
        
        return result


class ACFDecayRatio(RollingPrimitive):
    """
    ACF decay ratio.
    
    The autocorrelation of absolute returns at a long lag divided by its
    value at a short lag, as a direct measure of decay speed.
    
    Stylized Fact: Slow decay of autocorrelation in absolute returns (6)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None,
                 short_lag: int = 1,
                 long_lag: int = 10):
        super().__init__(
            config=config or PrimitiveConfig(),
            name=f"acf_decay_ratio_{short_lag}_{long_lag}",
            description=f"ACF decay ratio: ACF({long_lag}) / ACF({short_lag})"
        )
        self.short_lag = short_lag
        self.long_lag = long_lag
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute ACF decay ratio.
        
        Args:
            data: Input data (returns)
            window: Window size
            
        Returns:
            DataFrame with ACF decay ratio values
        """
        window = self._get_window(window)
        min_periods = self._get_min_periods(None)
        
        def acf_decay_ratio(x):
            """Compute ACF decay ratio for a single window."""
            if len(x) < self.long_lag + 1:
                return np.nan
            
            abs_x = np.abs(x)
            
            # Compute ACF at short lag
            if len(abs_x) < self.short_lag + 1:
                return np.nan
            acf_short = np.corrcoef(abs_x[:-self.short_lag], abs_x[self.short_lag:])[0, 1]
            
            # Compute ACF at long lag
            if len(abs_x) < self.long_lag + 1:
                return np.nan
            acf_long = np.corrcoef(abs_x[:-self.long_lag], abs_x[self.long_lag:])[0, 1]
            
            # Compute ratio
            if acf_short == 0:
                return np.nan
            
            return acf_long / acf_short
        
        result = data.rolling(
            window=window,
            min_periods=min_periods
        ).apply(acf_decay_ratio, raw=True)
        
        return result
