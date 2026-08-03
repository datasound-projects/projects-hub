"""
Hurst exponent primitives for stylized fact 6.

This module implements Hurst exponent estimation methods for detecting
persistence and mean reversion in time series.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
import logging

from .base import Primitive, PrimitiveConfig, RollingPrimitive

logger = logging.getLogger(__name__)


@dataclass
class HurstConfig(PrimitiveConfig):
    """Configuration for Hurst exponent estimation."""
    method: str = "rs"  # 'rs' for rescaled range, 'dfa' for detrended fluctuation analysis
    min_window: int = 10
    max_window: int = 100
    num_windows: int = 10


class HurstExponentRS(RollingPrimitive):
    """
    Hurst exponent via Rescaled Range (R/S) method.
    
    Estimates the Hurst exponent on rolling windows. Values above 0.5 indicate
    persistence (long memory), below 0.5 indicate antipersistence (mean reversion).
    
    Stylized Fact: Slow decay of autocorrelation in absolute returns (6)
    """
    
    def __init__(self, 
                 config: Optional[HurstConfig] = None):
        super().__init__(
            config=config or HurstConfig(),
            name="hurst_rs",
            description="Hurst exponent via Rescaled Range method"
        )
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute Hurst exponent using R/S method.
        
        Args:
            data: Input data (returns)
            window: Window size
            
        Returns:
            DataFrame with Hurst exponent values
        """
        window = self._get_window(window)
        min_periods = self._get_min_periods(None)
        
        def hurst_rs(x):
            """Compute Hurst exponent using R/S method for a single window."""
            if len(x) < min_periods:
                return np.nan
            
            # Remove mean
            x = x - np.mean(x)
            
            # Compute cumulative sum (Brownian bridge)
            cumsum = np.cumsum(x)
            
            # Compute range
            range_vals = np.max(cumsum) - np.min(cumsum)
            
            # Compute standard deviation
            std = np.std(x)
            
            if std <= 0:
                return np.nan
            
            # Rescaled range
            rs = range_vals / std
            
            # Hurst exponent approximation for a single window
            # For a single window, we use the expected relationship
            # E[R/S] = c * n^H, where c is a constant
            # For n=window, we can estimate H
            n = len(x)
            if n <= 1:
                return np.nan
            
            # Use the approximation: H = log(R/S) / log(n) + 0.5
            # This is a simplified estimation for a single window
            h = np.log(rs) / np.log(n) + 0.5
            
            # Constrain to reasonable range
            return np.clip(h, 0.0, 1.0)
        
        result = data.rolling(
            window=window,
            min_periods=min_periods
        ).apply(hurst_rs, raw=True)
        
        return result


class HurstExponentDFA(RollingPrimitive):
    """
    Hurst exponent via Detrended Fluctuation Analysis (DFA).
    
    More robust method for estimating Hurst exponent, especially for
    non-stationary time series.
    
    Stylized Fact: Slow decay of autocorrelation in absolute returns (6)
    """
    
    def __init__(self, 
                 config: Optional[HurstConfig] = None):
        super().__init__(
            config=config or HurstConfig(method="dfa"),
            name="hurst_dfa",
            description="Hurst exponent via Detrended Fluctuation Analysis"
        )
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute Hurst exponent using DFA method.
        
        Args:
            data: Input data (returns)
            window: Window size
            
        Returns:
            DataFrame with Hurst exponent values
        """
        window = self._get_window(window)
        min_periods = self._get_min_periods(None)
        
        def hurst_dfa(x):
            """Compute Hurst exponent using DFA for a single window."""
            if len(x) < min_periods:
                return np.nan
            
            # Convert to numpy array
            x = np.asarray(x)
            
            # Remove mean
            x = x - np.mean(x)
            
            # Compute cumulative sum
            cumsum = np.cumsum(x)
            
            # Define range of window sizes
            n = len(x)
            min_w = max(4, int(n * 0.1))
            max_w = min(int(n * 0.5), 100)
            window_sizes = np.arange(min_w, max_w, max(1, (max_w - min_w) // 5))
            
            if len(window_sizes) < 2:
                return np.nan
            
            # Compute fluctuation function F(n) for each window size
            f_n = []
            for w in window_sizes:
                # Split into windows of size w
                num_windows = n // w
                if num_windows < 2:
                    continue
                
                fluctuations = []
                for i in range(num_windows):
                    start = i * w
                    end = start + w
                    window_data = cumsum[start:end]
                    
                    # Fit linear trend
                    x_vals = np.arange(w)
                    coeffs = np.polyfit(x_vals, window_data, 1)
                    trend = np.polyval(coeffs, x_vals)
                    
                    # Compute RMS of detrended data
                    detrended = window_data - trend
                    rms = np.sqrt(np.mean(detrended ** 2))
                    fluctuations.append(rms)
                
                if fluctuations:
                    f_n.append((w, np.mean(fluctuations)))
            
            if len(f_n) < 2:
                return np.nan
            
            # Extract window sizes and F(n) values
            window_sizes_arr = np.array([x[0] for x in f_n])
            f_n_arr = np.array([x[1] for x in f_n])
            
            # Fit power law: F(n) = n^H
            # In log-log space: log(F(n)) = H * log(n)
            log_w = np.log(window_sizes_arr)
            log_f = np.log(f_n_arr)
            
            # Linear regression
            A = np.vstack([log_w, np.ones(len(log_w))]).T
            H, _ = np.linalg.lstsq(A, log_f, rcond=None)[0]
            
            return np.clip(H, 0.0, 1.0)
        
        result = data.rolling(
            window=window,
            min_periods=min_periods
        ).apply(hurst_dfa, raw=True)
        
        return result


class HurstExponent(RollingPrimitive):
    """
    Unified Hurst exponent primitive.
    
    Uses the method specified in the configuration (default: R/S).
    
    Stylized Fact: Slow decay of autocorrelation in absolute returns (6)
    """
    
    def __init__(self, 
                 config: Optional[HurstConfig] = None,
                 method: str = "rs"):
        super().__init__(
            config=config or HurstConfig(method=method),
            name=f"hurst_{method}",
            description=f"Hurst exponent via {method.upper()} method"
        )
        self.method = method
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute Hurst exponent using the specified method.
        
        Args:
            data: Input data (returns)
            window: Window size
            
        Returns:
            DataFrame with Hurst exponent values
        """
        if self.method == "rs" or self.method == "rescaled_range":
            hurst_calc = HurstExponentRS(self.config)
        elif self.method == "dfa" or self.method == "detrended_fluctuation":
            hurst_calc = HurstExponentDFA(self.config)
        else:
            raise ValueError(f"Unknown Hurst method: {self.method}")
        
        return hurst_calc.compute(data, window)
