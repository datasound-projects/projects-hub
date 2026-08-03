"""
Volatility-related primitives for stylized facts 1-4.

This module implements primitives related to:
- Fat tails (Stylized Fact 1)
- Volatility clustering (Stylized Fact 2)
- Leverage effect (Stylized Fact 3)
- Volume-volatility correlation (Stylized Fact 4)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass
import logging

from .base import Primitive, PrimitiveConfig, RollingPrimitive

logger = logging.getLogger(__name__)


# ============================================================================
# Fat Tails (Stylized Fact 1)
# ============================================================================

@dataclass
class KurtosisConfig(PrimitiveConfig):
    """Configuration for rolling kurtosis."""
    fisher: bool = True  # Fisher's definition (excess kurtosis)
    bias: bool = False   # Bias correction


class RollingKurtosis(RollingPrimitive):
    """
    Rolling kurtosis of returns.
    
    Measures the fourth standardized moment, indicating fat tails.
    Positive values indicate heavier tails than normal distribution.
    
    Stylized Fact: Excess kurtosis / fat tails (1)
    """
    
    def __init__(self, 
                 config: Optional[KurtosisConfig] = None):
        super().__init__(
            config=config or KurtosisConfig(),
            name="rolling_kurtosis",
            description="Rolling kurtosis of returns (fat tails indicator)"
        )
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute rolling kurtosis.
        
        Args:
            data: Input data (returns)
            window: Window size
            
        Returns:
            DataFrame with rolling kurtosis values
        """
        window = self._get_window(window)
        config = self.config
        
        if isinstance(config, PrimitiveConfig):
            config = KurtosisConfig(**config.__dict__)
        
        # Compute rolling kurtosis
        result = data.rolling(
            window=window,
            min_periods=self._get_min_periods(None)
        ).kurt(fisher=config.fisher, bias=config.bias)
        
        return result


class HillEstimator(RollingPrimitive):
    """
    Hill estimator for power-law tail index.
    
    Approximates the tail index by fitting a Pareto distribution to the largest
    absolute observations within the window. Lower values indicate fatter tails
    and higher extreme risk.
    
    Stylized Fact: Excess kurtosis / fat tails (1)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None,
                 threshold_quantile: float = 0.95):
        super().__init__(
            config=config or PrimitiveConfig(),
            name="hill_estimator",
            description="Hill estimator for power-law tail index"
        )
        self.threshold_quantile = threshold_quantile
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute Hill estimator for tail index.
        
        Args:
            data: Input data (returns)
            window: Window size
            
        Returns:
            DataFrame with Hill estimator values
        """
        window = self._get_window(window)
        min_periods = self._get_min_periods(None)
        
        def hill_estimator(x):
            """Compute Hill estimator for a single window."""
            if len(x) < min_periods:
                return np.nan
            
            # Get absolute values and sort in descending order
            abs_x = np.abs(x)
            abs_x_sorted = np.sort(abs_x)[::-1]
            
            # Use top (1 - threshold_quantile) * 100% of observations
            n_tail = max(10, int(len(abs_x_sorted) * (1 - self.threshold_quantile)))
            tail = abs_x_sorted[:n_tail]
            
            if len(tail) < 2:
                return np.nan
            
            # Hill estimator: 1 / mean(log(tail / min_tail))
            log_ratios = np.log(tail / tail[-1])
            
            if np.sum(log_ratios) <= 0:
                return np.nan
            
            return 1 / np.mean(log_ratios)
        
        result = data.rolling(
            window=window,
            min_periods=min_periods
        ).apply(hill_estimator, raw=True)
        
        return result


class ExtremeRatio(RollingPrimitive):
    """
    Extreme ratio: fraction of observations exceeding 2 standard deviations.
    
    Computes the fraction of observations exceeding 2 standard deviations,
    divided by the fraction expected under normality. Values greater than 1
    confirm fat-tailed behavior.
    
    Stylized Fact: Excess kurtosis / fat tails (1)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None,
                 threshold: float = 2.0):
        super().__init__(
            config=config or PrimitiveConfig(),
            name="extreme_ratio",
            description="Ratio of extreme observations to normal expectation"
        )
        self.threshold = threshold
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute extreme ratio.
        
        Args:
            data: Input data (returns)
            window: Window size
            
        Returns:
            DataFrame with extreme ratio values
        """
        window = self._get_window(window)
        min_periods = self._get_min_periods(None)
        
        def extreme_ratio(x):
            """Compute extreme ratio for a single window."""
            if len(x) < min_periods:
                return np.nan
            
            # Compute standard deviation
            std = np.std(x)
            if std <= 0:
                return np.nan
            
            # Count observations exceeding threshold * std
            abs_x = np.abs(x)
            extreme_count = np.sum(abs_x > self.threshold * std)
            
            # Expected count under normality (two-tailed)
            from scipy.stats import norm
            expected_prob = 2 * (1 - norm.cdf(self.threshold))
            expected_count = len(x) * expected_prob
            
            if expected_count <= 0:
                return np.nan
            
            return extreme_count / expected_count
        
        result = data.rolling(
            window=window,
            min_periods=min_periods
        ).apply(extreme_ratio, raw=True)
        
        return result


# ============================================================================
# Volatility Clustering (Stylized Fact 2)
# ============================================================================

class RollingVolatility(RollingPrimitive):
    """
    Rolling volatility (standard deviation) of returns.
    
    Basic measure of volatility over a rolling window.
    
    Stylized Fact: Volatility clustering (2)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None):
        super().__init__(
            config=config or PrimitiveConfig(),
            name="rolling_volatility",
            description="Rolling standard deviation of returns"
        )
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute rolling volatility.
        
        Args:
            data: Input data (returns)
            window: Window size
            
        Returns:
            DataFrame with rolling volatility values
        """
        window = self._get_window(window)
        return self._rolling_std(data, window)


class VolatilityClustering(RollingPrimitive):
    """
    Autocorrelation of absolute returns.
    
    Direct measure of volatility clustering: large price movements tend to be
    followed by large movements, and small by small.
    
    Stylized Fact: Volatility clustering (2)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None,
                 lag: int = 1):
        super().__init__(
            config=config or PrimitiveConfig(),
            name=f"volatility_clustering_lag_{lag}",
            description=f"Autocorrelation of absolute returns at lag {lag}"
        )
        self.lag = lag
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute autocorrelation of absolute returns.
        
        Args:
            data: Input data (returns)
            window: Window size
            
        Returns:
            DataFrame with autocorrelation values
        """
        window = self._get_window(window)
        min_periods = self._get_min_periods(None)
        
        # Compute absolute returns
        abs_returns = data.abs()
        
        # Compute rolling autocorrelation at specified lag
        def autocorr(x):
            if len(x) < self.lag + 1:
                return np.nan
            return np.corrcoef(x[:-self.lag], x[self.lag:])[0, 1]
        
        result = abs_returns.rolling(
            window=window,
            min_periods=min_periods
        ).apply(autocorr, raw=True)
        
        return result


class VolatilityRatio(RollingPrimitive):
    """
    Ratio of short-term to long-term realized volatility.
    
    Captures whether the market is currently in a high- or low-volatility
    regime relative to its recent history.
    
    Stylized Fact: Volatility clustering (2)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None,
                 short_window: int = 5,
                 long_window: int = 60):
        super().__init__(
            config=config or PrimitiveConfig(),
            name=f"volatility_ratio_{short_window}_{long_window}",
            description=f"Ratio of {short_window}-day to {long_window}-day volatility"
        )
        self.short_window = short_window
        self.long_window = long_window
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute volatility ratio.
        
        Args:
            data: Input data (returns)
            window: Window size (not used, uses short_window and long_window)
            
        Returns:
            DataFrame with volatility ratio values
        """
        # Compute short-term volatility
        short_vol = self._rolling_std(data, self.short_window)
        
        # Compute long-term volatility
        long_vol = self._rolling_std(data, self.long_window)
        
        # Compute ratio
        result = short_vol / long_vol
        
        return result


class EWMVolatility(RollingPrimitive):
    """
    Exponentially weighted moving average volatility.
    
    Provides faster-reacting volatility estimates with different half-lives.
    
    Stylized Fact: Volatility clustering (2)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None,
                 halflife: Optional[float] = None,
                 alpha: Optional[float] = None):
        super().__init__(
            config=config or PrimitiveConfig(),
            name=f"ewm_volatility_{halflife or 'auto'}",
            description="Exponentially weighted moving average volatility"
        )
        self.halflife = halflife
        self.alpha = alpha
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute EWM volatility.
        
        Args:
            data: Input data (returns)
            window: Window size (not used for EWM)
            
        Returns:
            DataFrame with EWM volatility values
        """
        # Compute EWM standard deviation
        result = data.ewm(
            halflife=self.halflife,
            alpha=self.alpha,
            min_periods=self._get_min_periods(None)
        ).std()
        
        return result


class VolatilityOfVolatility(RollingPrimitive):
    """
    Volatility of volatility.
    
    Measures the standard deviation of rolling standard deviations, capturing
    whether volatility is stable or erratic.
    
    Stylized Fact: Volatility clustering (2)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None,
                 inner_window: int = 20,
                 outer_window: int = 60):
        super().__init__(
            config=config or PrimitiveConfig(),
            name=f"volatility_of_volatility_{inner_window}_{outer_window}",
            description="Standard deviation of rolling volatility"
        )
        self.inner_window = inner_window
        self.outer_window = outer_window
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute volatility of volatility.
        
        Args:
            data: Input data (returns)
            window: Window size (not used, uses inner_window and outer_window)
            
        Returns:
            DataFrame with volatility of volatility values
        """
        # Compute inner volatility
        inner_vol = self._rolling_std(data, self.inner_window)
        
        # Compute volatility of the inner volatility
        result = self._rolling_std(inner_vol, self.outer_window)
        
        return result


# ============================================================================
# Leverage Effect (Stylized Fact 3)
# ============================================================================

class DownsideUpsideVolatilityRatio(RollingPrimitive):
    """
    Downside-upside volatility ratio.
    
    Computes the standard deviation of negative returns divided by the standard
    deviation of positive returns. Values greater than 1 indicate leverage effect.
    
    Stylized Fact: Leverage effect (3)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None):
        super().__init__(
            config=config or PrimitiveConfig(),
            name="downside_upside_vol_ratio",
            description="Ratio of downside to upside volatility"
        )
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute downside-upside volatility ratio.
        
        Args:
            data: Input data (returns)
            window: Window size
            
        Returns:
            DataFrame with downside-upside volatility ratio values
        """
        window = self._get_window(window)
        min_periods = self._get_min_periods(None)
        
        # Separate positive and negative returns
        positive_returns = data.clip(lower=0)
        negative_returns = data.clip(upper=0).abs()
        
        # Compute rolling standard deviations
        pos_vol = self._rolling_std(positive_returns, window)
        neg_vol = self._rolling_std(negative_returns, window)
        
        # Compute ratio (avoid division by zero)
        result = neg_vol / pos_vol.replace(0, np.nan)
        
        return result


class LeverageCorrelation(RollingPrimitive):
    """
    Rolling correlation between returns and changes in volatility.
    
    Should be negative when the leverage effect is present (negative returns
    increase future volatility more than positive returns).
    
    Stylized Fact: Leverage effect (3)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None,
                 vol_window: int = 20):
        super().__init__(
            config=config or PrimitiveConfig(),
            name="leverage_correlation",
            description="Correlation between returns and volatility changes"
        )
        self.vol_window = vol_window
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute leverage correlation.
        
        Args:
            data: Input data (returns)
            window: Window size
            
        Returns:
            DataFrame with leverage correlation values
        """
        window = self._get_window(window)
        min_periods = self._get_min_periods(None)
        
        # Compute volatility
        vol = self._rolling_std(data, self.vol_window)
        
        # Compute changes in volatility
        vol_change = vol.diff()
        
        # Compute rolling correlation between returns and volatility changes
        result = self._rolling_corr(data, vol_change, window)
        
        return result


class GJRAsymmetry(RollingPrimitive):
    """
    GJR asymmetry measure.
    
    Inspired by the GJR-GARCH model, computes the ratio of the mean squared
    negative return to the mean squared positive return.
    
    Stylized Fact: Leverage effect (3)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None):
        super().__init__(
            config=config or PrimitiveConfig(),
            name="gjr_asymmetry",
            description="GJR asymmetry measure (mean squared negative / positive returns)"
        )
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute GJR asymmetry.
        
        Args:
            data: Input data (returns)
            window: Window size
            
        Returns:
            DataFrame with GJR asymmetry values
        """
        window = self._get_window(window)
        min_periods = self._get_min_periods(None)
        
        # Separate positive and negative returns
        positive_returns = data.clip(lower=0)
        negative_returns = data.clip(upper=0).abs()
        
        # Compute squared returns
        pos_squared = positive_returns ** 2
        neg_squared = negative_returns ** 2
        
        # Compute rolling means
        pos_mean = self._rolling_mean(pos_squared, window)
        neg_mean = self._rolling_mean(neg_squared, window)
        
        # Compute ratio (avoid division by zero)
        result = neg_mean / pos_mean.replace(0, np.nan)
        
        return result


# ============================================================================
# Volume-Volatility Correlation (Stylized Fact 4)
# ============================================================================

class VolumeVolatilityCorrelation(RollingPrimitive):
    """
    Rolling correlation between absolute returns and volume.
    
    Trading volume tends to increase with absolute returns, consistent with
    models of information arrival.
    
    Stylized Fact: Volume-volatility correlation (4)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None):
        super().__init__(
            config=config or PrimitiveConfig(),
            name="volume_volatility_corr",
            description="Correlation between absolute returns and volume"
        )
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute volume-volatility correlation.
        
        Args:
            data: Input data (must contain 'abs_return' and 'volume' columns)
            window: Window size
            
        Returns:
            DataFrame with correlation values
        """
        window = self._get_window(window)
        
        if 'abs_return' not in data.columns or 'volume' not in data.columns:
            raise ValueError("Data must contain 'abs_return' and 'volume' columns")
        
        abs_returns = data['abs_return']
        volume = data['volume']
        
        result = self._rolling_corr(abs_returns, volume, window)
        
        return result


class RelativeVolume(RollingPrimitive):
    """
    Relative volume.
    
    Current volume divided by its rolling average.
    
    Stylized Fact: Volume-volatility correlation (4)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None):
        super().__init__(
            config=config or PrimitiveConfig(),
            name="relative_volume",
            description="Current volume divided by rolling average volume"
        )
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute relative volume.
        
        Args:
            data: Input data (must contain 'volume' column)
            window: Window size
            
        Returns:
            DataFrame with relative volume values
        """
        window = self._get_window(window)
        
        if 'volume' not in data.columns:
            raise ValueError("Data must contain 'volume' column")
        
        volume = data['volume']
        vol_mean = self._rolling_mean(volume, window)
        
        result = volume / vol_mean.replace(0, np.nan)
        
        return result


class VolumeSurprise(RollingPrimitive):
    """
    Volume surprise.
    
    Current volume as a fraction of its rolling maximum.
    
    Stylized Fact: Volume-volatility correlation (4)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None):
        super().__init__(
            config=config or PrimitiveConfig(),
            name="volume_surprise",
            description="Current volume as fraction of rolling maximum"
        )
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute volume surprise.
        
        Args:
            data: Input data (must contain 'volume' column)
            window: Window size
            
        Returns:
            DataFrame with volume surprise values
        """
        window = self._get_window(window)
        
        if 'volume' not in data.columns:
            raise ValueError("Data must contain 'volume' column")
        
        volume = data['volume']
        vol_max = volume.rolling(window=window, min_periods=self._get_min_periods(None)).max()
        
        result = volume / vol_max.replace(0, np.nan)
        
        return result


class VolumeWeightedVolatility(RollingPrimitive):
    """
    Volume-weighted volatility.
    
    Weights return variance by concurrent trading activity, providing a more
    informative measure of "true" volatility than equal-weighted alternatives.
    
    Stylized Fact: Volume-volatility correlation (4)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None):
        super().__init__(
            config=config or PrimitiveConfig(),
            name="volume_weighted_volatility",
            description="Volume-weighted volatility"
        )
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute volume-weighted volatility.
        
        Args:
            data: Input data (must contain 'return' and 'volume' columns)
            window: Window size
            
        Returns:
            DataFrame with volume-weighted volatility values
        """
        window = self._get_window(window)
        
        if 'log_return' not in data.columns and 'return' not in data.columns:
            raise ValueError("Data must contain 'log_return' or 'return' column")
        
        if 'volume' not in data.columns:
            raise ValueError("Data must contain 'volume' column")
        
        returns = data.get('log_return', data['return'])
        volume = data['volume']
        
        # Compute volume-weighted returns
        weighted_returns = returns * volume
        
        # Compute rolling standard deviation of weighted returns
        result = self._rolling_std(weighted_returns, window)
        
        return result


class DollarVolume(RollingPrimitive):
    """
    Dollar volume.
    
    Trading volume multiplied by typical price.
    
    Stylized Fact: Volume-volatility correlation (4)
    """
    
    def __init__(self, 
                 config: Optional[PrimitiveConfig] = None):
        super().__init__(
            config=config or PrimitiveConfig(),
            name="dollar_volume",
            description="Volume multiplied by typical price"
        )
    
    def compute(self, 
                data: pd.DataFrame,
                window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute dollar volume.
        
        Args:
            data: Input data (must contain 'volume' and 'typical_price' columns)
            window: Window size (not used)
            
        Returns:
            DataFrame with dollar volume values
        """
        if 'volume' not in data.columns:
            raise ValueError("Data must contain 'volume' column")
        
        if 'typical_price' not in data.columns:
            if 'close' in data.columns:
                # Use close price as approximation
                typical_price = data['close']
            else:
                raise ValueError("Data must contain 'typical_price' or 'close' column")
        else:
            typical_price = data['typical_price']
        
        result = data['volume'] * typical_price
        
        return result.to_frame() if isinstance(result, pd.Series) else result
