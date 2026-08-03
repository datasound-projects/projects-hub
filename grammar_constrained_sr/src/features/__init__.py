"""
Feature computation module for stylized facts primitives.

This module implements all 15 stylized fact primitives described in the methodology,
organized by the phenomenon they encode.
"""

from .base import Primitive, PrimitiveConfig, FeatureConfig
from .volatility import (
    RollingVolatility, 
    VolatilityClustering, 
    VolatilityOfVolatility,
    EWMVolatility,
    VolatilityRatio
)
from .autocorrelation import (
    RollingAutocorrelation,
    AutocorrelationDecay,
    VarianceRatio,
    HurstExponent,
    ACFDecayRatio
)
from .spectral import (
    SpectralEntropy,
    SpectralPowerRatio,
    PermutationEntropy
)
from .wavelet import (
    WaveletEnergy,
    WaveletNoiseToSignalRatio,
    WaveletDecomposition
)
from .hilbert import (
    HilbertAmplitude,
    HilbertPhase,
    HilbertTransform
)
from .hurst import HurstExponentRS, HurstExponentDFA
from .momentum import (
    RollingReturn,
    CumulativeReturn,
    TrendStrength,
    TrendLinearity,
    JegadeeshTitmanMomentum
)
from .volume import (
    VolumeVolatilityCorrelation,
    RelativeVolume,
    VolumeSurprise,
    VolumeWeightedVolatility,
    DollarVolume
)
from .leverage import (
    DownsideUpsideVolatilityRatio,
    LeverageCorrelation,
    GJRAsymmetry
)
from .skewness import (
    RollingSkewness,
    DownsideFrequency,
    GainLossAsymmetry
)
from .mean_reversion import (
    HodrickPrescottCycle,
    ZeroCrossingRate,
    NormalizedDeviation,
    MeanReversionSignal
)
from .lead_lag import (
    RollingBeta,
    HouMoskowitzDelay,
    LeadLagEffect
)
from .taylor import (
    TaylorRatio,
    OptimalTaylorExponent
)
from .efficiency import (
    MarketEfficiency,
    AR1RSquared,
    AdaptiveMarketHypothesis
)
from .factory import FeatureFactory

__all__ = [
    # Base classes
    'Primitive', 'PrimitiveConfig', 'FeatureConfig',
    # Volatility
    'RollingVolatility', 'VolatilityClustering', 'VolatilityOfVolatility',
    'EWMVolatility', 'VolatilityRatio',
    # Autocorrelation
    'RollingAutocorrelation', 'AutocorrelationDecay', 'VarianceRatio',
    'HurstExponent', 'ACFDecayRatio',
    # Spectral
    'SpectralEntropy', 'SpectralPowerRatio', 'PermutationEntropy',
    # Wavelet
    'WaveletEnergy', 'WaveletNoiseToSignalRatio', 'WaveletDecomposition',
    # Hilbert
    'HilbertAmplitude', 'HilbertPhase', 'HilbertTransform',
    # Hurst
    'HurstExponentRS', 'HurstExponentDFA',
    # Momentum
    'RollingReturn', 'CumulativeReturn', 'TrendStrength', 'TrendLinearity',
    'JegadeeshTitmanMomentum',
    # Volume
    'VolumeVolatilityCorrelation', 'RelativeVolume', 'VolumeSurprise',
    'VolumeWeightedVolatility', 'DollarVolume',
    # Leverage
    'DownsideUpsideVolatilityRatio', 'LeverageCorrelation', 'GJRAsymmetry',
    # Skewness
    'RollingSkewness', 'DownsideFrequency', 'GainLossAsymmetry',
    # Mean Reversion
    'HodrickPrescottCycle', 'ZeroCrossingRate', 'NormalizedDeviation',
    'MeanReversionSignal',
    # Lead-Lag
    'RollingBeta', 'HouMoskowitzDelay', 'LeadLagEffect',
    # Taylor
    'TaylorRatio', 'OptimalTaylorExponent',
    # Efficiency
    'MarketEfficiency', 'AR1RSquared', 'AdaptiveMarketHypothesis',
    # Factory
    'FeatureFactory'
]
