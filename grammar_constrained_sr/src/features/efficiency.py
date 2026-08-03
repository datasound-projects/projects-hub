"""
Market efficiency primitives for stylized fact 15.

This module implements primitives related to:
- Time-varying market efficiency (Stylized Fact 15)
"""

import pandas as pd
import numpy as np
from typing import Optional
from dataclasses import dataclass
import logging

from .base import Primitive, PrimitiveConfig, RollingPrimitive

logger = logging.getLogger(__name__)


# ============================================================================
# Time-Varying Market Efficiency (Stylized Fact 15)
# ============================================================================

class MarketEfficiency(RollingPrimitive):
    """
    Market efficiency measure.
    
    Combines multiple measures of market efficiency into a single indicator.
    
    Stylized Fact: Time-varying market efficiency (15)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None):
        super().__init__(
            config=config or PrimitiveConfig(),
            name="market_efficiency",
            description="Combined market efficiency measure"
        )
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute market efficiency.
        
        Args:
            data: Input data (returns)
            window: Window size
            
        Returns:
            DataFrame with market efficiency values
        """
        window = self._get_window(window)
        
        # Compute spectral entropy (lower = more predictable = less efficient)
        spectral_entropy = self._compute_spectral_entropy(data, window)
        
        # Compute permutation entropy (lower = more regular = less efficient)
        perm_entropy = self._compute_permutation_entropy(data, window)
        
        # Compute AR(1) R-squared (higher = more predictable = less efficient)
        ar1_rsquared = self._compute_ar1_rsquared(data, window)
        
        # Combine measures (invert spectral and permutation entropy since lower = less efficient)
        # Normalize each measure to [0, 1] range
        efficiency = (1 - spectral_entropy + 1 - perm_entropy + ar1_rsquared) / 3
        
        return efficiency
    
    def _compute_spectral_entropy(self, 
                                   data: pd.DataFrame,
                                   window: int) -> pd.DataFrame:
        """Compute spectral entropy."""
        min_periods = self._get_min_periods(None)
        
        def spectral_entropy(x):
            if len(x) < min_periods:
                return np.nan
            
            # Compute power spectral density
            psd = np.abs(np.fft.fft(x)) ** 2
            psd_normalized = psd / np.sum(psd)
            psd_normalized = psd_normalized[psd_normalized > 0]
            
            if len(psd_normalized) == 0:
                return np.nan
            
            return -np.sum(psd_normalized * np.log(psd_normalized))
        
        result = data.rolling(
            window=window,
            min_periods=min_periods
        ).apply(spectral_entropy, raw=True)
        
        # Normalize to [0, 1] range
        result = (result - result.min()) / (result.max() - result.min() + 1e-10)
        
        return result
    
    def _compute_permutation_entropy(self, 
                                     data: pd.DataFrame,
                                     window: int) -> pd.DataFrame:
        """Compute permutation entropy."""
        min_periods = self._get_min_periods(None)
        m = 3  # Embedding dimension
        
        def perm_entropy(x):
            if len(x) < min_periods or len(x) < m:
                return np.nan
            
            # Create embedding matrix
            num_vectors = len(x) - m + 1
            permutations = []
            
            for i in range(num_vectors):
                pattern = x[i:i+m]
                perm = np.argsort(np.argsort(pattern))
                permutations.append(tuple(perm))
            
            from collections import Counter
            perm_counts = Counter(permutations)
            total_perms = len(permutations)
            
            if total_perms == 0:
                return np.nan
            
            probs = np.array([count / total_perms for count in perm_counts.values()])
            probs = probs[probs > 0]
            
            if len(probs) == 0:
                return np.nan
            
            entropy = -np.sum(probs * np.log(probs))
            
            # Normalize by maximum entropy
            import math
            max_entropy = np.log(math.factorial(m))
            if max_entropy > 0:
                entropy /= max_entropy
            
            return entropy
        
        result = data.rolling(
            window=window,
            min_periods=min_periods
        ).apply(perm_entropy, raw=True)
        
        # Normalize to [0, 1] range
        result = (result - result.min()) / (result.max() - result.min() + 1e-10)
        
        return result
    
    def _compute_ar1_rsquared(self, 
                             data: pd.DataFrame,
                             window: int) -> pd.DataFrame:
        """Compute AR(1) R-squared."""
        min_periods = self._get_min_periods(None)
        
        def ar1_rsquared(x):
            if len(x) < min_periods or len(x) < 2:
                return np.nan
            
            # Fit AR(1): x_t = phi * x_{t-1} + epsilon
            x_prev = x[:-1]
            x_curr = x[1:]
            
            # Compute correlation
            corr = np.corrcoef(x_prev, x_curr)[0, 1]
            
            # R-squared is correlation squared for simple linear regression
            return corr ** 2
        
        result = data.rolling(
            window=window,
            min_periods=min_periods
        ).apply(ar1_rsquared, raw=True)
        
        return result


class AR1RSquared(RollingPrimitive):
    """
    R-squared of a rolling AR(1) model.
    
    The fraction of return variance explained by the simplest linear forecast.
    Higher values indicate more predictable returns (less efficient markets).
    
    Stylized Fact: Time-varying market efficiency (15)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None):
        super().__init__(
            config=config or PrimitiveConfig(),
            name="ar1_rsquared",
            description="R-squared of rolling AR(1) model"
        )
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute AR(1) R-squared.
        
        Args:
            data: Input data (returns)
            window: Window size
            
        Returns:
            DataFrame with AR(1) R-squared values
        """
        window = self._get_window(window)
        min_periods = self._get_min_periods(None)
        
        def ar1_rsquared(x):
            """Compute AR(1) R-squared for a single window."""
            if len(x) < min_periods or len(x) < 2:
                return np.nan
            
            # Fit AR(1): x_t = phi * x_{t-1} + epsilon
            x_prev = x[:-1]
            x_curr = x[1:]
            
            # Compute correlation
            corr = np.corrcoef(x_prev, x_curr)[0, 1]
            
            # R-squared is correlation squared for simple linear regression
            return corr ** 2
        
        result = data.rolling(
            window=window,
            min_periods=min_periods
        ).apply(ar1_rsquared, raw=True)
        
        return result


class AdaptiveMarketHypothesis(RollingPrimitive):
    """
    Adaptive Market Hypothesis measure.
    
    Implements Lo's (2004) Adaptive Markets Hypothesis by measuring
    time-varying efficiency through multiple indicators.
    
    Stylized Fact: Time-varying market efficiency (15)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None):
        super().__init__(
            config=config or PrimitiveConfig(),
            name="adaptive_market_hypothesis",
            description="Adaptive Market Hypothesis measure"
        )
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute AMH measure.
        
        Args:
            data: Input data (returns)
            window: Window size
            
        Returns:
            DataFrame with AMH values
        """
        window = self._get_window(window)
        
        # Compute spectral entropy
        spectral_entropy = self._compute_spectral_entropy(data, window)
        
        # Compute AR(1) R-squared
        ar1_rsquared = self._compute_ar1_rsquared(data, window)
        
        # Compute volatility (inverse of efficiency)
        volatility = self._rolling_std(data, window)
        volatility = (volatility - volatility.min()) / (volatility.max() - volatility.min() + 1e-10)
        
        # AMH: higher values indicate less efficient markets
        # Combine: higher spectral entropy = more efficient
        #         higher AR(1) R-squared = less efficient
        #         higher volatility = less efficient
        amh = (1 - spectral_entropy + ar1_rsquared + volatility) / 3
        
        return amh
    
    def _compute_spectral_entropy(self, 
                                   data: pd.DataFrame,
                                   window: int) -> pd.DataFrame:
        """Compute spectral entropy."""
        min_periods = self._get_min_periods(None)
        
        def spectral_entropy(x):
            if len(x) < min_periods:
                return np.nan
            
            psd = np.abs(np.fft.fft(x)) ** 2
            psd_normalized = psd / np.sum(psd)
            psd_normalized = psd_normalized[psd_normalized > 0]
            
            if len(psd_normalized) == 0:
                return np.nan
            
            return -np.sum(psd_normalized * np.log(psd_normalized))
        
        result = data.rolling(
            window=window,
            min_periods=min_periods
        ).apply(spectral_entropy, raw=True)
        
        # Normalize
        result = (result - result.min()) / (result.max() - result.min() + 1e-10)
        
        return result
    
    def _compute_ar1_rsquared(self, 
                             data: pd.DataFrame,
                             window: int) -> pd.DataFrame:
        """Compute AR(1) R-squared."""
        min_periods = self._get_min_periods(None)
        
        def ar1_rsquared(x):
            if len(x) < min_periods or len(x) < 2:
                return np.nan
            
            x_prev = x[:-1]
            x_curr = x[1:]
            corr = np.corrcoef(x_prev, x_curr)[0, 1]
            return corr ** 2
        
        result = data.rolling(
            window=window,
            min_periods=min_periods
        ).apply(ar1_rsquared, raw=True)
        
        return result
