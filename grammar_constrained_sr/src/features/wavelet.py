"""
Wavelet decomposition primitives for stylized fact 13.

This module implements primitives related to:
- Coarse-fine volatility asymmetry (Stylized Fact 13)
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
import logging

try:
    import pywt
    WAVELET_AVAILABLE = True
except ImportError:
    WAVELET_AVAILABLE = False
    logger.warning("PyWavelets not available. Wavelet primitives will not work.")

from .base import Primitive, PrimitiveConfig, RollingPrimitive

logger = logging.getLogger(__name__)


@dataclass
class WaveletConfig(PrimitiveConfig):
    """Configuration for wavelet decomposition."""
    wavelet: str = "db4"  # Daubechies 4
    level: int = 3
    mode: str = "symmetric"


class WaveletDecomposition(RollingPrimitive):
    """
    Wavelet decomposition of time series.
    
    Decomposes a signal into frequency-specific components, enabling
    multi-scale volatility analysis.
    
    Stylized Fact: Coarse-fine volatility asymmetry (13)
    """
    
    def __init__(self, 
                 config: Optional[WaveletConfig] = None):
        if not WAVELET_AVAILABLE:
            raise ImportError("PyWavelets is required for wavelet primitives. Install with: pip install pywavelets")
        
        super().__init__(
            config=config or WaveletConfig(),
            name="wavelet_decomposition",
            description="Wavelet decomposition of time series"
        )
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute wavelet decomposition.
        
        Args:
            data: Input data (returns)
            window: Window size
            
        Returns:
            DataFrame with wavelet coefficients
        """
        window = self._get_window(window)
        min_periods = self._get_min_periods(None)
        
        def wavelet_decompose(x):
            """Perform wavelet decomposition on a single window."""
            if len(x) < min_periods:
                return pd.Series([np.nan] * self.config.level, 
                               index=[f"wavelet_level_{i}" for i in range(self.config.level)])
            
            # Pad to power of 2 if needed
            n = len(x)
            n_padded = 1
            while n_padded < n:
                n_padded *= 2
            
            if n_padded > n:
                x_padded = np.pad(x, (0, n_padded - n), mode=self.config.mode)
            else:
                x_padded = x
            
            # Perform wavelet decomposition
            coeffs = pywt.wavedec(x_padded, self.config.wavelet, level=self.config.level)
            
            # Extract detail coefficients (approximation is coeffs[0])
            detail_coeffs = coeffs[1:]  # Skip approximation
            
            # Compute energy (variance) of each detail level
            energies = [np.var(coeff) for coeff in detail_coeffs]
            
            # Pad or truncate to expected number of levels
            while len(energies) < self.config.level:
                energies.append(np.nan)
            energies = energies[:self.config.level]
            
            return pd.Series(energies, 
                           index=[f"wavelet_level_{i}" for i in range(self.config.level)])
        
        # Apply wavelet decomposition to each column
        results = data.rolling(
            window=window,
            min_periods=min_periods
        ).apply(wavelet_decompose, raw=True)
        
        # Convert to DataFrame
        result_df = pd.concat(results, axis=1)
        
        return result_df


class WaveletEnergy(RollingPrimitive):
    """
    Wavelet energy at specific decomposition levels.
    
    Computes the variance of detail coefficients at specific levels,
    corresponding to different time scales.
    
    Stylized Fact: Coarse-fine volatility asymmetry (13)
    """
    
    def __init__(self, 
                 config: Optional[WaveletConfig] = None,
                 levels: List[int] = [1, 2, 3]):
        if not WAVELET_AVAILABLE:
            raise ImportError("PyWavelets is required for wavelet primitives. Install with: pip install pywavelets")
        
        super().__init__(
            config=config or WaveletConfig(),
            name=f"wavelet_energy_levels_{levels}",
            description=f"Wavelet energy at levels {levels}"
        )
        self.levels = levels
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute wavelet energy at specified levels.
        
        Args:
            data: Input data (returns)
            window: Window size
            
        Returns:
            DataFrame with wavelet energy values
        """
        window = self._get_window(window)
        min_periods = self._get_min_periods(None)
        
        def wavelet_energy(x):
            """Compute wavelet energy at specified levels."""
            if len(x) < min_periods:
                return pd.Series([np.nan] * len(self.levels), 
                               index=[f"wavelet_energy_level_{i}" for i in self.levels])
            
            # Pad to power of 2 if needed
            n = len(x)
            n_padded = 1
            while n_padded < n:
                n_padded *= 2
            
            if n_padded > n:
                x_padded = np.pad(x, (0, n_padded - n), mode=self.config.mode)
            else:
                x_padded = x
            
            # Perform wavelet decomposition
            max_level = max(self.levels)
            coeffs = pywt.wavedec(x_padded, self.config.wavelet, level=max_level)
            
            # Extract detail coefficients
            detail_coeffs = coeffs[1:]  # Skip approximation
            
            # Compute energy for requested levels
            energies = []
            for level in self.levels:
                if level <= len(detail_coeffs):
                    energy = np.var(detail_coeffs[level - 1])
                else:
                    energy = np.nan
                energies.append(energy)
            
            return pd.Series(energies, 
                           index=[f"wavelet_energy_level_{i}" for i in self.levels])
        
        # Apply to each column
        results = data.rolling(
            window=window,
            min_periods=min_periods
        ).apply(wavelet_energy, raw=True)
        
        # Convert to DataFrame
        result_df = pd.concat(results, axis=1)
        
        return result_df


class WaveletNoiseToSignalRatio(RollingPrimitive):
    """
    Wavelet noise-to-signal ratio.
    
    Computes the energy at level 1 (high frequency) divided by energy at
    level 3 (low frequency), providing a measure of noise relative to signal.
    
    Stylized Fact: Coarse-fine volatility asymmetry (13)
    """
    
    def __init__(self, 
                 config: Optional[WaveletConfig] = None,
                 noise_level: int = 1,
                 signal_level: int = 3):
        if not WAVELET_AVAILABLE:
            raise ImportError("PyWavelets is required for wavelet primitives. Install with: pip install pywavelets")
        
        super().__init__(
            config=config or WaveletConfig(),
            name=f"wavelet_noise_to_signal_{noise_level}_{signal_level}",
            description=f"Wavelet noise-to-signal ratio: level {noise_level} / level {signal_level}"
        )
        self.noise_level = noise_level
        self.signal_level = signal_level
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute wavelet noise-to-signal ratio.
        
        Args:
            data: Input data (returns)
            window: Window size
            
        Returns:
            DataFrame with noise-to-signal ratio values
        """
        window = self._get_window(window)
        min_periods = self._get_min_periods(None)
        
        def noise_to_signal(x):
            """Compute noise-to-signal ratio."""
            if len(x) < min_periods:
                return np.nan
            
            # Pad to power of 2 if needed
            n = len(x)
            n_padded = 1
            while n_padded < n:
                n_padded *= 2
            
            if n_padded > n:
                x_padded = np.pad(x, (0, n_padded - n), mode=self.config.mode)
            else:
                x_padded = x
            
            # Perform wavelet decomposition
            max_level = max(self.noise_level, self.signal_level)
            coeffs = pywt.wavedec(x_padded, self.config.wavelet, level=max_level)
            
            # Extract detail coefficients
            detail_coeffs = coeffs[1:]  # Skip approximation
            
            # Compute energies
            noise_energy = np.var(detail_coeffs[self.noise_level - 1]) if self.noise_level <= len(detail_coeffs) else np.nan
            signal_energy = np.var(detail_coeffs[self.signal_level - 1]) if self.signal_level <= len(detail_coeffs) else np.nan
            
            if signal_energy <= 0:
                return np.nan
            
            return noise_energy / signal_energy
        
        result = data.rolling(
            window=window,
            min_periods=min_periods
        ).apply(noise_to_signal, raw=True)
        
        return result
