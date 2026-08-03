"""
Hilbert transform primitives for stylized fact 10.

This module implements primitives related to:
- Mean reversion (Stylized Fact 10)
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
import logging

from .base import Primitive, PrimitiveConfig, RollingPrimitive

logger = logging.getLogger(__name__)


class HilbertTransform:
    """
    Hilbert transform utilities.
    
    Provides static methods for computing Hilbert transform and its components.
    """
    
    @staticmethod
    def hilbert(x: np.ndarray) -> np.ndarray:
        """
        Compute the Hilbert transform of a signal.
        
        Args:
            x: Input signal
            
        Returns:
            Hilbert transform of the signal
        """
        n = len(x)
        
        # Pad to next power of 2
        n_padded = 1
        while n_padded < n:
            n_padded *= 2
        
        if n_padded > n:
            x_padded = np.pad(x, (0, n_padded - n), mode='reflect')
        else:
            x_padded = x
        
        # Compute FFT
        fft_x = np.fft.fft(x_padded)
        
        # Create Hilbert transform filter
        h = np.zeros(n_padded)
        h[0] = 1
        h[n_padded // 2] = 1
        for i in range(1, n_padded // 2):
            h[i] = 2
            h[n_padded - i] = 2
        
        # Apply filter in frequency domain
        hilbert_fft = fft_x * h
        
        # Inverse FFT
        hilbert_x = np.fft.ifft(hilbert_fft).real
        
        # Return only the original length
        return hilbert_x[:n]
    
    @staticmethod
    def instantaneous_amplitude(x: np.ndarray) -> np.ndarray:
        """
        Compute instantaneous amplitude (envelope) of a signal.
        
        Args:
            x: Input signal
            
        Returns:
            Instantaneous amplitude
        """
        # Compute Hilbert transform
        hilbert_x = HilbertTransform.hilbert(x)
        
        # Compute analytic signal
        analytic = x + 1j * hilbert_x
        
        # Compute amplitude (envelope)
        amplitude = np.abs(analytic)
        
        return amplitude
    
    @staticmethod
    def instantaneous_phase(x: np.ndarray) -> np.ndarray:
        """
        Compute instantaneous phase of a signal.
        
        Args:
            x: Input signal
            
        Returns:
            Instantaneous phase in radians
        """
        # Compute Hilbert transform
        hilbert_x = HilbertTransform.hilbert(x)
        
        # Compute analytic signal
        analytic = x + 1j * hilbert_x
        
        # Compute phase
        phase = np.angle(analytic)
        
        return phase


class HilbertAmplitude(RollingPrimitive):
    """
    Rolling Hilbert transform amplitude.
    
    Measures the strength of the current cycle in returns. High amplitude
    indicates strong cyclical behavior.
    
    Stylized Fact: Mean reversion (10)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None):
        super().__init__(
            config=config or PrimitiveConfig(),
            name="hilbert_amplitude",
            description="Instantaneous amplitude from Hilbert transform"
        )
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute rolling Hilbert amplitude.
        
        Args:
            data: Input data (returns)
            window: Window size
            
        Returns:
            DataFrame with Hilbert amplitude values
        """
        window = self._get_window(window)
        min_periods = self._get_min_periods(None)
        
        def hilbert_amplitude(x):
            """Compute Hilbert amplitude for a single window."""
            if len(x) < min_periods:
                return np.nan
            
            # Demean the signal
            x_demeaned = x - np.mean(x)
            
            # Compute instantaneous amplitude
            amplitude = HilbertTransform.instantaneous_amplitude(x_demeaned)
            
            # Return the last value (most recent)
            return amplitude[-1]
        
        result = data.rolling(
            window=window,
            min_periods=min_periods
        ).apply(hilbert_amplitude, raw=True)
        
        return result


class HilbertPhase(RollingPrimitive):
    """
    Rolling Hilbert transform phase.
    
    Identifies the position within the current cycle (trough, ascending,
    peak, or descending). Phase values:
    - 0: trough
    - π/2: ascending zero crossing
    - π: peak
    - -π/2: descending zero crossing
    
    Stylized Fact: Mean reversion (10)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None):
        super().__init__(
            config=config or PrimitiveConfig(),
            name="hilbert_phase",
            description="Instantaneous phase from Hilbert transform"
        )
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute rolling Hilbert phase.
        
        Args:
            data: Input data (returns)
            window: Window size
            
        Returns:
            DataFrame with Hilbert phase values
        """
        window = self._get_window(window)
        min_periods = self._get_min_periods(None)
        
        def hilbert_phase(x):
            """Compute Hilbert phase for a single window."""
            if len(x) < min_periods:
                return np.nan
            
            # Demean the signal
            x_demeaned = x - np.mean(x)
            
            # Compute instantaneous phase
            phase = HilbertTransform.instantaneous_phase(x_demeaned)
            
            # Return the last value (most recent)
            return phase[-1]
        
        result = data.rolling(
            window=window,
            min_periods=min_periods
        ).apply(hilbert_phase, raw=True)
        
        return result


class HilbertTransformPrimitive(RollingPrimitive):
    """
    Combined Hilbert transform primitive.
    
    Computes both amplitude and phase for use in alpha discovery.
    
    Stylized Fact: Mean reversion (10)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None):
        super().__init__(
            config=config or PrimitiveConfig(),
            name="hilbert_transform",
            description="Hilbert transform (amplitude and phase)"
        )
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute Hilbert transform (amplitude and phase).
        
        Args:
            data: Input data (returns)
            window: Window size
            
        Returns:
            DataFrame with both amplitude and phase
        """
        window = self._get_window(window)
        min_periods = self._get_min_periods(None)
        
        def hilbert_transform(x):
            """Compute Hilbert transform for a single window."""
            if len(x) < min_periods:
                return pd.Series({'hilbert_amplitude': np.nan, 'hilbert_phase': np.nan})
            
            # Demean the signal
            x_demeaned = x - np.mean(x)
            
            # Compute amplitude and phase
            amplitude = HilbertTransform.instantaneous_amplitude(x_demeaned)
            phase = HilbertTransform.instantaneous_phase(x_demeaned)
            
            return pd.Series({
                'hilbert_amplitude': amplitude[-1],
                'hilbert_phase': phase[-1]
            })
        
        # Apply to each column
        results = data.rolling(
            window=window,
            min_periods=min_periods
        ).apply(hilbert_transform, raw=True)
        
        # Convert to DataFrame
        result_df = pd.concat(results, axis=1)
        
        return result_df
