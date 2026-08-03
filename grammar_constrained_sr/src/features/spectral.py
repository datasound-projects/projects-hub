"""
Spectral analysis primitives for stylized facts 14-15.

This module implements primitives related to:
- Taylor effect (Stylized Fact 14)
- Time-varying market efficiency (Stylized Fact 15)
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
import logging

from .base import Primitive, PrimitiveConfig, RollingPrimitive

logger = logging.getLogger(__name__)


# ============================================================================
# Spectral Analysis Primitives
# ============================================================================

class SpectralEntropy(RollingPrimitive):
    """
    Spectral entropy.
    
    The Shannon entropy of the normalized power spectral density. Low values
    indicate concentrated spectral power and thus greater predictability.
    
    Stylized Fact: Time-varying market efficiency (15)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None,
                 nfft: int = 256):
        super().__init__(
            config=config or PrimitiveConfig(),
            name="spectral_entropy",
            description="Shannon entropy of normalized power spectral density"
        )
        self.nfft = nfft
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute spectral entropy.
        
        Args:
            data: Input data (returns)
            window: Window size
            
        Returns:
            DataFrame with spectral entropy values
        """
        window = self._get_window(window)
        min_periods = self._get_min_periods(None)
        
        def spectral_entropy(x):
            """Compute spectral entropy for a single window."""
            if len(x) < min_periods:
                return np.nan
            
            # Compute power spectral density
            psd = np.abs(np.fft.fft(x, n=self.nfft)) ** 2
            
            # Normalize
            psd_normalized = psd / np.sum(psd)
            
            # Filter out zero values
            psd_normalized = psd_normalized[psd_normalized > 0]
            
            if len(psd_normalized) == 0:
                return np.nan
            
            # Compute Shannon entropy
            entropy = -np.sum(psd_normalized * np.log(psd_normalized))
            
            return entropy
        
        result = data.rolling(
            window=window,
            min_periods=min_periods
        ).apply(spectral_entropy, raw=True)
        
        return result


class SpectralPowerRatio(RollingPrimitive):
    """
    Spectral power ratio.
    
    The energy in low frequencies divided by energy in high frequencies.
    
    Stylized Fact: Coarse-fine volatility asymmetry (13)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None,
                 nfft: int = 256,
                 low_freq_ratio: float = 0.25):
        super().__init__(
            config=config or PrimitiveConfig(),
            name="spectral_power_ratio",
            description="Energy in low frequencies divided by energy in high frequencies"
        )
        self.nfft = nfft
        self.low_freq_ratio = low_freq_ratio
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute spectral power ratio.
        
        Args:
            data: Input data (returns)
            window: Window size
            
        Returns:
            DataFrame with spectral power ratio values
        """
        window = self._get_window(window)
        min_periods = self._get_min_periods(None)
        
        def spectral_power_ratio(x):
            """Compute spectral power ratio for a single window."""
            if len(x) < min_periods:
                return np.nan
            
            # Compute power spectral density
            psd = np.abs(np.fft.fft(x, n=self.nfft)) ** 2
            
            # Split into low and high frequencies
            split_point = int(self.low_freq_ratio * self.nfft)
            
            low_freq_energy = np.sum(psd[:split_point])
            high_freq_energy = np.sum(psd[split_point:])
            
            if high_freq_energy <= 0:
                return np.nan
            
            return low_freq_energy / high_freq_energy
        
        result = data.rolling(
            window=window,
            min_periods=min_periods
        ).apply(spectral_power_ratio, raw=True)
        
        return result


class PermutationEntropy(RollingPrimitive):
    """
    Permutation entropy.
    
    An ordinal-pattern-based complexity measure robust to outliers. Low
    values indicate more regular, predictable patterns.
    
    Stylized Fact: Time-varying market efficiency (15)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None,
                 m: int = 3,
                 delay: int = 1):
        """
        Initialize permutation entropy.
        
        Args:
            config: Primitive configuration
            m: Embedding dimension (order of permutation)
            delay: Delay between samples
        """
        super().__init__(
            config=config or PrimitiveConfig(),
            name=f"permutation_entropy_m{m}_d{delay}",
            description=f"Permutation entropy with m={m}, delay={delay}"
        )
        self.m = m
        self.delay = delay
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute permutation entropy.
        
        Args:
            data: Input data (returns)
            window: Window size
            
        Returns:
            DataFrame with permutation entropy values
        """
        window = self._get_window(window)
        min_periods = self._get_min_periods(None)
        
        def permutation_entropy(x):
            """Compute permutation entropy for a single window."""
            if len(x) < min_periods or len(x) < self.m * self.delay:
                return np.nan
            
            # Create embedding matrix
            n = len(x)
            num_vectors = n - (self.m - 1) * self.delay
            
            if num_vectors < 1:
                return np.nan
            
            # Generate all possible permutations
            permutations = []
            for i in range(num_vectors):
                indices = [i + j * self.delay for j in range(self.m)]
                pattern = x[indices]
                # Get the permutation (rank order)
                perm = np.argsort(np.argsort(pattern))
                permutations.append(tuple(perm))
            
            # Count frequency of each permutation
            from collections import Counter
            perm_counts = Counter(permutations)
            total_perms = len(permutations)
            
            # Compute probabilities
            probs = np.array([count / total_perms for count in perm_counts.values()])
            probs = probs[probs > 0]  # Filter out zero probabilities
            
            if len(probs) == 0:
                return np.nan
            
            # Compute Shannon entropy
            entropy = -np.sum(probs * np.log(probs))
            
            # Normalize by maximum possible entropy
            max_entropy = np.log(self.m!)
            if max_entropy > 0:
                entropy /= max_entropy
            
            return entropy
        
        # Compute factorial for normalization
        import math
        self.m_factorial = math.factorial(self.m)
        
        result = data.rolling(
            window=window,
            min_periods=min_periods
        ).apply(permutation_entropy, raw=True)
        
        return result


# ============================================================================
# Taylor Effect (Stylized Fact 14)
# ============================================================================

class TaylorRatio(RollingPrimitive):
    """
    Taylor ratio.
    
    The autocorrelation of |r| divided by autocorrelation of r^2.
    The Taylor effect states that this ratio is maximized at d=1, not d=2.
    
    Stylized Fact: Taylor effect (14)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None,
                 lag: int = 1):
        super().__init__(
            config=config or PrimitiveConfig(),
            name=f"taylor_ratio_lag_{lag}",
            description=f"Taylor ratio: ACF(|r|) / ACF(r^2) at lag {lag}"
        )
        self.lag = lag
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute Taylor ratio.
        
        Args:
            data: Input data (returns)
            window: Window size
            
        Returns:
            DataFrame with Taylor ratio values
        """
        window = self._get_window(window)
        min_periods = self._get_min_periods(None)
        
        def taylor_ratio(x):
            """Compute Taylor ratio for a single window."""
            if len(x) < self.lag + 1:
                return np.nan
            
            abs_r = np.abs(x)
            r_squared = x ** 2
            
            # Compute ACF of |r| at lag
            if len(abs_r) < self.lag + 1:
                return np.nan
            acf_abs = np.corrcoef(abs_r[:-self.lag], abs_r[self.lag:])[0, 1]
            
            # Compute ACF of r^2 at lag
            if len(r_squared) < self.lag + 1:
                return np.nan
            acf_sq = np.corrcoef(r_squared[:-self.lag], r_squared[self.lag:])[0, 1]
            
            # Compute ratio
            if acf_sq == 0:
                return np.nan
            
            return acf_abs / acf_sq
        
        result = data.rolling(
            window=window,
            min_periods=min_periods
        ).apply(taylor_ratio, raw=True)
        
        return result


class OptimalTaylorExponent(RollingPrimitive):
    """
    Optimal Taylor exponent.
    
    The value of d that maximizes autocorrelation of |r|^d, estimated by grid search.
    Deviation from d=1 may indicate unusual microstructure dynamics.
    
    Stylized Fact: Taylor effect (14)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None,
                 d_range: Tuple[float, float] = (0.1, 2.0),
                 d_steps: int = 20):
        super().__init__(
            config=config or PrimitiveConfig(),
            name="optimal_taylor_exponent",
            description="Optimal d that maximizes autocorrelation of |r|^d"
        )
        self.d_range = d_range
        self.d_steps = d_steps
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute optimal Taylor exponent.
        
        Args:
            data: Input data (returns)
            window: Window size
            
        Returns:
            DataFrame with optimal Taylor exponent values
        """
        window = self._get_window(window)
        min_periods = self._get_min_periods(None)
        
        def optimal_taylor_exponent(x):
            """Find optimal d that maximizes autocorrelation of |r|^d."""
            if len(x) < min_periods:
                return np.nan
            
            abs_r = np.abs(x)
            d_values = np.linspace(self.d_range[0], self.d_range[1], self.d_steps)
            
            max_acf = -np.inf
            best_d = 1.0
            
            for d in d_values:
                # Compute |r|^d
                r_d = abs_r ** d
                
                # Compute autocorrelation at lag 1
                if len(r_d) < 2:
                    continue
                acf = np.corrcoef(r_d[:-1], r_d[1:])[0, 1]
                
                if acf > max_acf:
                    max_acf = acf
                    best_d = d
            
            return best_d
        
        result = data.rolling(
            window=window,
            min_periods=min_periods
        ).apply(optimal_taylor_exponent, raw=True)
        
        return result
