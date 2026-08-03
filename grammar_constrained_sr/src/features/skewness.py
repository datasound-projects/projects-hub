"""
Skewness-related primitives for stylized facts 7-9.

This module implements primitives related to:
- Negative skewness (Stylized Fact 7)
- Aggregational Gaussianity (Stylized Fact 8)
- Gain/loss asymmetry (Stylized Fact 9)
"""

import pandas as pd
import numpy as np
from typing import Optional
from dataclasses import dataclass
import logging

from .base import Primitive, PrimitiveConfig, RollingPrimitive

logger = logging.getLogger(__name__)


# ============================================================================
# Negative Skewness (Stylized Fact 7)
# ============================================================================

class RollingSkewness(RollingPrimitive):
    """
    Rolling skewness of returns.
    
    Measures the asymmetry of the return distribution. Negative skewness
    indicates that large downward movements are more frequent than large
    upward movements.
    
    Stylized Fact: Negative skewness (7)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None,
                 bias: bool = False):
        super().__init__(
            config=config or PrimitiveConfig(),
            name="rolling_skewness",
            description="Rolling skewness of returns"
        )
        self.bias = bias
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute rolling skewness.
        
        Args:
            data: Input data (returns)
            window: Window size
            
        Returns:
            DataFrame with rolling skewness values
        """
        window = self._get_window(window)
        
        result = data.rolling(
            window=window,
            min_periods=self._get_min_periods(None)
        ).skew(bias=self.bias)
        
        return result


class DownsideFrequency(RollingPrimitive):
    """
    Downside frequency.
    
    The ratio of returns below a negative threshold to returns above the
    corresponding positive threshold.
    
    Stylized Fact: Negative skewness (7)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None,
                 threshold: float = 0.01):
        super().__init__(
            config=config or PrimitiveConfig(),
            name=f"downside_frequency_{threshold}",
            description=f"Ratio of returns < -{threshold} to returns > {threshold}"
        )
        self.threshold = threshold
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute downside frequency.
        
        Args:
            data: Input data (returns)
            window: Window size
            
        Returns:
            DataFrame with downside frequency values
        """
        window = self._get_window(window)
        min_periods = self._get_min_periods(None)
        
        def downside_freq(x):
            """Compute downside frequency for a single window."""
            if len(x) < min_periods:
                return np.nan
            
            # Count returns below negative threshold
            down_count = np.sum(x < -self.threshold)
            
            # Count returns above positive threshold
            up_count = np.sum(x > self.threshold)
            
            if up_count == 0:
                return np.nan
            
            return down_count / up_count
        
        result = data.rolling(
            window=window,
            min_periods=min_periods
        ).apply(downside_freq, raw=True)
        
        return result


# ============================================================================
# Aggregational Gaussianity (Stylized Fact 8)
# ============================================================================

class KurtosisRatio(RollingPrimitive):
    """
    Kurtosis ratio.
    
    Excess kurtosis of daily returns divided by excess kurtosis of aggregated
    (e.g., five-day) returns. A high ratio indicates that non-Gaussianity is
    concentrated at high frequencies, consistent with microstructure effects
    or event-driven dynamics.
    
    Stylized Fact: Aggregational Gaussianity (8)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None,
                 daily_window: int = 20,
                 aggregated_window: int = 100):
        super().__init__(
            config=config or PrimitiveConfig(),
            name=f"kurtosis_ratio_{daily_window}_{aggregated_window}",
            description=f"Kurtosis ratio: daily ({daily_window}) / aggregated ({aggregated_window})"
        )
        self.daily_window = daily_window
        self.aggregated_window = aggregated_window
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute kurtosis ratio.
        
        Args:
            data: Input data (returns)
            window: Window size (not used)
            
        Returns:
            DataFrame with kurtosis ratio values
        """
        # Compute daily kurtosis
        daily_kurtosis = data.rolling(
            window=self.daily_window,
            min_periods=self._get_min_periods(None)
        ).kurt(fisher=True)
        
        # Compute aggregated returns (5-day)
        aggregated_returns = (1 + data).rolling(window=5).apply(
            lambda x: np.prod(x) - 1, raw=True
        )
        
        # Compute aggregated kurtosis
        aggregated_kurtosis = aggregated_returns.rolling(
            window=self.aggregated_window,
            min_periods=self._get_min_periods(None)
        ).kurt(fisher=True)
        
        # Compute ratio
        result = daily_kurtosis / aggregated_kurtosis.replace(0, np.nan)
        
        return result


# ============================================================================
# Gain/Loss Asymmetry (Stylized Fact 9)
# ============================================================================

class UpDownRunRatio(RollingPrimitive):
    """
    Up-down run ratio.
    
    Average duration of consecutive positive-return days divided by average
    duration of consecutive negative-return days. Values above 1 indicate
    that gains develop slowly while losses arrive abruptly.
    
    Stylized Fact: Gain/loss asymmetry (9)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None):
        super().__init__(
            config=config or PrimitiveConfig(),
            name="up_down_run_ratio",
            description="Average duration of positive runs / average duration of negative runs"
        )
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute up-down run ratio.
        
        Args:
            data: Input data (returns)
            window: Window size
            
        Returns:
            DataFrame with up-down run ratio values
        """
        window = self._get_window(window)
        min_periods = self._get_min_periods(None)
        
        def up_down_run_ratio(x):
            """Compute up-down run ratio for a single window."""
            if len(x) < min_periods:
                return np.nan
            
            # Identify positive and negative runs
            signs = np.sign(x)
            
            # Find run starts and ends
            run_starts = np.where(np.diff(signs, prepend=signs[0]) != 0)[0]
            
            if len(run_starts) < 2:
                return np.nan
            
            # Compute run lengths
            run_lengths = np.diff(np.concatenate([[0], run_starts, [len(x)]]))
            
            # Separate positive and negative runs
            pos_runs = []
            neg_runs = []
            
            for i, length in zip(run_starts, run_lengths):
                if signs[i] > 0:
                    pos_runs.append(length)
                elif signs[i] < 0:
                    neg_runs.append(length)
            
            if len(pos_runs) == 0 or len(neg_runs) == 0:
                return np.nan
            
            return np.mean(pos_runs) / np.mean(neg_runs)
        
        result = data.rolling(
            window=window,
            min_periods=min_periods
        ).apply(up_down_run_ratio, raw=True)
        
        return result


class ConditionalMeanRatio(RollingPrimitive):
    """
    Conditional mean ratio.
    
    Mean positive return divided by the absolute value of mean negative return.
    
    Stylized Fact: Gain/loss asymmetry (9)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None):
        super().__init__(
            config=config or PrimitiveConfig(),
            name="conditional_mean_ratio",
            description="Mean positive return / |mean negative return|"
        )
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute conditional mean ratio.
        
        Args:
            data: Input data (returns)
            window: Window size
            
        Returns:
            DataFrame with conditional mean ratio values
        """
        window = self._get_window(window)
        
        # Compute mean of positive returns
        pos_mean = data.clip(lower=0).rolling(
            window=window,
            min_periods=self._get_min_periods(None)
        ).mean()
        
        # Compute mean of negative returns
        neg_mean = data.clip(upper=0).rolling(
            window=window,
            min_periods=self._get_min_periods(None)
        ).mean()
        
        # Compute ratio
        result = pos_mean / neg_mean.abs().replace(0, np.nan)
        
        return result


class GainLossAsymmetry(RollingPrimitive):
    """
    Combined gain/loss asymmetry measure.
    
    Combines multiple measures of gain/loss asymmetry into a single primitive.
    
    Stylized Fact: Gain/loss asymmetry (9)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None):
        super().__init__(
            config=config or PrimitiveConfig(),
            name="gain_loss_asymmetry",
            description="Combined gain/loss asymmetry measure"
        )
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute gain/loss asymmetry.
        
        Args:
            data: Input data (returns)
            window: Window size
            
        Returns:
            DataFrame with gain/loss asymmetry values
        """
        window = self._get_window(window)
        
        # Compute skewness (negative = more negative skew)
        skewness = data.rolling(
            window=window,
            min_periods=self._get_min_periods(None)
        ).skew()
        
        # Compute conditional mean ratio
        pos_mean = data.clip(lower=0).rolling(
            window=window,
            min_periods=self._get_min_periods(None)
        ).mean()
        neg_mean = data.clip(upper=0).rolling(
            window=window,
            min_periods=self._get_min_periods(None)
        ).mean()
        mean_ratio = pos_mean / neg_mean.abs().replace(0, np.nan)
        
        # Combine measures (simple average)
        result = (skewness + (1 - mean_ratio)) / 2
        
        return result
