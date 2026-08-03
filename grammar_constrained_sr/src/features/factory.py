"""
Feature factory for creating and managing stylized fact primitives.

This module provides a centralized way to create, configure, and compute
all the primitives for the grammar-constrained symbolic regression.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
import logging

from .base import Primitive, PrimitiveConfig, FeatureConfig
from .volatility import (
    RollingKurtosis, HillEstimator, ExtremeRatio,
    RollingVolatility, VolatilityClustering, VolatilityRatio,
    EWMVolatility, VolatilityOfVolatility,
    DownsideUpsideVolatilityRatio, LeverageCorrelation, GJRAsymmetry,
    VolumeVolatilityCorrelation, RelativeVolume, VolumeSurprise,
    VolumeWeightedVolatility, DollarVolume
)
from .autocorrelation import (
    RollingAutocorrelation, VarianceRatio,
    AutocorrelationDecay, ACFDecayRatio
)
from .hurst import HurstExponentRS, HurstExponentDFA, HurstExponent
from .spectral import (
    SpectralEntropy, SpectralPowerRatio, PermutationEntropy,
    TaylorRatio, OptimalTaylorExponent
)
from .wavelet import (
    WaveletDecomposition, WaveletEnergy, WaveletNoiseToSignalRatio
)
from .hilbert import HilbertAmplitude, HilbertPhase, HilbertTransformPrimitive
from .momentum import (
    RollingReturn, CumulativeReturn, TrendStrength,
    TrendLinearity, JegadeeshTitmanMomentum
)
from .skewness import (
    RollingSkewness, DownsideFrequency, GainLossAsymmetry,
    KurtosisRatio, UpDownRunRatio, ConditionalMeanRatio
)
from .mean_reversion import (
    HodrickPrescottCycle, ZeroCrossingRate,
    NormalizedDeviation, MeanReversionSignal
)
from .lead_lag import RollingBeta, HouMoskowitzDelay, LeadLagEffect
from .efficiency import MarketEfficiency, AR1RSquared, AdaptiveMarketHypothesis

logger = logging.getLogger(__name__)


@dataclass
class StylizedFactGroup:
    """Group of primitives for a single stylized fact."""
    name: str
    fact_number: int
    primitives: List[Tuple[str, Primitive]]
    description: str


class FeatureFactory:
    """
    Factory for creating and managing stylized fact primitives.
    
    Provides methods to:
    - Create all primitives for a given configuration
    - Compute features for a panel of data
    - Group primitives by stylized fact
    - Manage primitive configuration
    """
    
    # Mapping of stylized fact numbers to their primitives
    STYLIZED_FACTS = {
        1: {
            'name': 'Fat Tails',
            'description': 'Excess kurtosis / fat tails',
            'primitives': [
                ('rolling_kurtosis', RollingKurtosis),
                ('hill_estimator', HillEstimator),
                ('extreme_ratio', ExtremeRatio),
            ]
        },
        2: {
            'name': 'Volatility Clustering',
            'description': 'ARCH effect - large movements followed by large movements',
            'primitives': [
                ('rolling_volatility', RollingVolatility),
                ('volatility_clustering', VolatilityClustering),
                ('volatility_ratio', VolatilityRatio),
                ('ewm_volatility', EWMVolatility),
                ('volatility_of_volatility', VolatilityOfVolatility),
            ]
        },
        3: {
            'name': 'Leverage Effect',
            'description': 'Negative returns increase future volatility more than positive returns',
            'primitives': [
                ('downside_upside_vol_ratio', DownsideUpsideVolatilityRatio),
                ('leverage_correlation', LeverageCorrelation),
                ('gjr_asymmetry', GJRAsymmetry),
            ]
        },
        4: {
            'name': 'Volume-Volatility Correlation',
            'description': 'Trading volume increases with absolute returns',
            'primitives': [
                ('volume_volatility_corr', VolumeVolatilityCorrelation),
                ('relative_volume', RelativeVolume),
                ('volume_surprise', VolumeSurprise),
                ('volume_weighted_volatility', VolumeWeightedVolatility),
                ('dollar_volume', DollarVolume),
            ]
        },
        5: {
            'name': 'Absence of Linear Autocorrelation',
            'description': 'Returns exhibit near-zero autocorrelation',
            'primitives': [
                ('rolling_autocorr', RollingAutocorrelation),
                ('variance_ratio', VarianceRatio),
            ]
        },
        6: {
            'name': 'Slow Decay of Autocorrelation in Absolute Returns',
            'description': 'Absolute returns exhibit long-range dependence',
            'primitives': [
                ('hurst_exponent', HurstExponent),
                ('hurst_rs', HurstExponentRS),
                ('hurst_dfa', HurstExponentDFA),
                ('autocorr_decay', AutocorrelationDecay),
                ('acf_decay_ratio', ACFDecayRatio),
            ]
        },
        7: {
            'name': 'Negative Skewness',
            'description': 'Large downward movements more frequent than upward',
            'primitives': [
                ('rolling_skewness', RollingSkewness),
                ('downside_frequency', DownsideFrequency),
            ]
        },
        8: {
            'name': 'Aggregational Gaussianity',
            'description': 'Return distribution converges to Gaussian as aggregation period increases',
            'primitives': [
                ('kurtosis_ratio', KurtosisRatio),
            ]
        },
        9: {
            'name': 'Gain/Loss Asymmetry',
            'description': 'Market drawdowns steep and rapid, recoveries gradual',
            'primitives': [
                ('up_down_run_ratio', UpDownRunRatio),
                ('conditional_mean_ratio', ConditionalMeanRatio),
                ('gain_loss_asymmetry', GainLossAsymmetry),
            ]
        },
        10: {
            'name': 'Mean Reversion',
            'description': 'At short horizons, returns tend to revert',
            'primitives': [
                ('hodrick_prescott_cycle', HodrickPrescottCycle),
                ('hilbert_amplitude', HilbertAmplitude),
                ('hilbert_phase', HilbertPhase),
                ('zero_crossing_rate', ZeroCrossingRate),
                ('normalized_deviation', NormalizedDeviation),
                ('mean_reversion_signal', MeanReversionSignal),
            ]
        },
        11: {
            'name': 'Momentum',
            'description': 'At intermediate horizons, assets that performed well continue to perform well',
            'primitives': [
                ('rolling_return', RollingReturn),
                ('cumulative_return', CumulativeReturn),
                ('trend_strength', TrendStrength),
                ('trend_linearity', TrendLinearity),
                ('jegadeesh_titman_momentum', JegadeeshTitmanMomentum),
            ]
        },
        12: {
            'name': 'Lead-Lag Effects',
            'description': 'Large, liquid assets react to information faster than small, illiquid ones',
            'primitives': [
                ('rolling_beta', RollingBeta),
                ('hou_moskowitz_delay', HouMoskowitzDelay),
                ('lead_lag_effect', LeadLagEffect),
            ]
        },
        13: {
            'name': 'Coarse-Fine Volatility Asymmetry',
            'description': 'Low-frequency volatility predicts high-frequency volatility better than reverse',
            'primitives': [
                ('wavelet_energy', WaveletEnergy),
                ('wavelet_noise_to_signal', WaveletNoiseToSignalRatio),
                ('spectral_power_ratio', SpectralPowerRatio),
            ]
        },
        14: {
            'name': 'Taylor Effect',
            'description': 'Autocorrelation of |r|^d is maximized at d approximately 1, not d=2',
            'primitives': [
                ('taylor_ratio', TaylorRatio),
                ('optimal_taylor_exponent', OptimalTaylorExponent),
            ]
        },
        15: {
            'name': 'Time-Varying Market Efficiency',
            'description': 'Market efficiency fluctuates over time (Adaptive Markets Hypothesis)',
            'primitives': [
                ('spectral_entropy', SpectralEntropy),
                ('permutation_entropy', PermutationEntropy),
                ('ar1_rsquared', AR1RSquared),
                ('market_efficiency', MarketEfficiency),
                ('adaptive_market_hypothesis', AdaptiveMarketHypothesis),
            ]
        }
    }
    
    def __init__(self, 
                 config: Optional[FeatureConfig] = None):
        """
        Initialize the FeatureFactory.
        
        Args:
            config: Feature configuration
        """
        self.config = config or FeatureConfig()
        self.primitives: Dict[str, Primitive] = {}
        self.stylized_fact_groups: Dict[int, StylizedFactGroup] = {}
        
        # Initialize all primitives
        self._initialize_primitives()
        
    def _initialize_primitives(self):
        """Initialize all primitives based on configuration."""
        logger.info("Initializing primitives...")
        
        # Create stylized fact groups
        for fact_num, fact_info in self.STYLIZED_FACTS.items():
            primitives = []
            
            for prim_name, prim_class in fact_info['primitives']:
                # Check if this primitive should be included
                if self._should_include_primitive(prim_name):
                    try:
                        # Create primitive instance
                        primitive = prim_class(PrimitiveConfig(
                            window=self.config.window_sizes[0],
                            min_periods=self.config.min_periods
                        ))
                        primitives.append((prim_name, primitive))
                        self.primitives[prim_name] = primitive
                        logger.debug(f"Created primitive: {prim_name}")
                    except Exception as e:
                        logger.warning(f"Failed to create primitive {prim_name}: {str(e)}")
            
            # Create stylized fact group
            self.stylized_fact_groups[fact_num] = StylizedFactGroup(
                name=fact_info['name'],
                fact_number=fact_num,
                primitives=primitives,
                description=fact_info['description']
            )
        
        logger.info(f"Initialized {len(self.primitives)} primitives across {len(self.stylized_fact_groups)} stylized facts")
    
    def _should_include_primitive(self, prim_name: str) -> bool:
        """Check if a primitive should be included based on configuration."""
        # Check for wavelet features
        if 'wavelet' in prim_name and not self.config.include_wavelet:
            return False
        
        # Check for Hilbert features
        if 'hilbert' in prim_name and not self.config.include_hilbert:
            return False
        
        # Check for spectral features
        if 'spectral' in prim_name and not self.config.include_spectral:
            return False
        
        return True
    
    def get_primitive(self, name: str) -> Optional[Primitive]:
        """Get a primitive by name."""
        return self.primitives.get(name)
    
    def get_stylized_fact_group(self, fact_number: int) -> Optional[StylizedFactGroup]:
        """Get a stylized fact group by number."""
        return self.stylized_fact_groups.get(fact_number)
    
    def get_all_primitives(self) -> Dict[str, Primitive]:
        """Get all primitives."""
        return self.primitives
    
    def get_primitives_by_fact(self, fact_number: int) -> List[Primitive]:
        """Get all primitives for a specific stylized fact."""
        group = self.get_stylized_fact_group(fact_number)
        if group:
            return [prim for _, prim in group.primitives]
        return []
    
    def compute_all_features(self, 
                            panel: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features for a panel of data.
        
        Args:
            panel: Input panel DataFrame with MultiIndex (symbol, date)
            
        Returns:
            DataFrame with all computed features
        """
        logger.info("Computing all features...")
        
        # Prepare input data
        # Extract returns from panel
        if 'log_return' in panel.columns:
            returns = panel['log_return'].unstack(level='symbol')
        elif 'return' in panel.columns:
            returns = panel['return'].unstack(level='symbol')
        else:
            raise ValueError("Panel must contain 'log_return' or 'return' column")
        
        # Also extract other needed columns
        volume = panel.get('volume', pd.DataFrame()).unstack(level='symbol') if 'volume' in panel.columns else None
        
        # Create feature DataFrame
        feature_data = {}
        
        # Compute each primitive
        for prim_name, primitive in self.primitives.items():
            try:
                logger.debug(f"Computing {prim_name}...")
                
                # Prepare input data for this primitive
                input_data = self._prepare_input_data(primitive, returns, volume, panel)
                
                # Compute for all window sizes
                if hasattr(primitive, 'compute_for_windows'):
                    window_results = primitive.compute_for_windows(
                        input_data, 
                        self.config.window_sizes
                    )
                    
                    for window, result in window_results.items():
                        # Store result for each window
                        for col in result.columns:
                            feature_name = f"{prim_name}_w{window}_{col}"
                            feature_data[feature_name] = result[col].stack()
                else:
                    # Compute with default window
                    result = primitive.compute(input_data, window=self.config.window_sizes[0])
                    
                    if isinstance(result, pd.DataFrame):
                        for col in result.columns:
                            feature_name = f"{prim_name}_{col}"
                            feature_data[feature_name] = result[col].stack()
                    else:
                        feature_name = prim_name
                        feature_data[feature_name] = result.stack()
                
            except Exception as e:
                logger.error(f"Failed to compute {prim_name}: {str(e)}")
        
        # Create final feature DataFrame
        features = pd.concat(feature_data, axis=1)
        features.index = panel.index
        
        logger.info(f"Computed {len(features.columns)} features")
        
        return features
    
    def _prepare_input_data(self, 
                           primitive: Primitive,
                           returns: pd.DataFrame,
                           volume: Optional[pd.DataFrame],
                           panel: pd.DataFrame) -> pd.DataFrame:
        """Prepare input data for a specific primitive."""
        # Start with returns
        input_data = returns.copy()
        
        # Add absolute returns (needed by many primitives)
        input_data['abs_return'] = returns.abs()
        
        # Add volume if available and needed
        if volume is not None:
            input_data['volume'] = volume
        
        # Add typical price if available
        if 'typical_price' in panel.columns:
            typical_price = panel['typical_price'].unstack(level='symbol')
            input_data['typical_price'] = typical_price
        
        # Add close price if available
        if 'close' in panel.columns:
            close = panel['close'].unstack(level='symbol')
            input_data['close'] = close
        
        return input_data
    
    def compute_features_for_fact(self, 
                                  fact_number: int,
                                  panel: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features for a specific stylized fact.
        
        Args:
            fact_number: Stylized fact number
            panel: Input panel DataFrame
            
        Returns:
            DataFrame with features for the specified stylized fact
        """
        group = self.get_stylized_fact_group(fact_number)
        if not group:
            raise ValueError(f"Stylized fact {fact_number} not found")
        
        logger.info(f"Computing features for stylized fact {fact_number}: {group.name}")
        
        # Prepare input data
        if 'log_return' in panel.columns:
            returns = panel['log_return'].unstack(level='symbol')
        elif 'return' in panel.columns:
            returns = panel['return'].unstack(level='symbol')
        else:
            raise ValueError("Panel must contain 'log_return' or 'return' column")
        
        volume = panel.get('volume', pd.DataFrame()).unstack(level='symbol') if 'volume' in panel.columns else None
        
        feature_data = {}
        
        for prim_name, primitive in group.primitives:
            try:
                input_data = self._prepare_input_data(primitive, returns, volume, panel)
                
                if hasattr(primitive, 'compute_for_windows'):
                    window_results = primitive.compute_for_windows(
                        input_data, 
                        self.config.window_sizes
                    )
                    
                    for window, result in window_results.items():
                        for col in result.columns:
                            feature_name = f"{prim_name}_w{window}_{col}"
                            feature_data[feature_name] = result[col].stack()
                else:
                    result = primitive.compute(input_data, window=self.config.window_sizes[0])
                    
                    if isinstance(result, pd.DataFrame):
                        for col in result.columns:
                            feature_name = f"{prim_name}_{col}"
                            feature_data[feature_name] = result[col].stack()
                    else:
                        feature_name = prim_name
                        feature_data[feature_name] = result.stack()
                
            except Exception as e:
                logger.error(f"Failed to compute {prim_name} for fact {fact_number}: {str(e)}")
        
        features = pd.concat(feature_data, axis=1)
        features.index = panel.index
        
        return features
    
    def get_feature_list(self) -> List[str]:
        """Get a list of all available feature names."""
        return list(self.primitives.keys())
    
    def get_stylized_fact_summary(self) -> Dict[int, Dict[str, Any]]:
        """Get a summary of all stylized facts and their primitives."""
        summary = {}
        
        for fact_num, group in self.stylized_fact_groups.items():
            summary[fact_num] = {
                'name': group.name,
                'description': group.description,
                'primitives': [prim_name for prim_name, _ in group.primitives],
                'count': len(group.primitives)
            }
        
        return summary
