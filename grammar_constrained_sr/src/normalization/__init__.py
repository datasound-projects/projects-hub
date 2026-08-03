"""
Cross-sectional normalization module.

This module implements Layer 2 of the architecture: cross-sectional normalization
of the primitives computed in Layer 1.
"""

from .cross_sectional import (
    CrossSectionalNormalizer,
    PercentileRankNormalizer,
    ZScoreNormalizer,
    MinMaxNormalizer,
    normalize_features,
    rank_normalize,
    zscore_normalize
)

__all__ = [
    'CrossSectionalNormalizer',
    'PercentileRankNormalizer',
    'ZScoreNormalizer',
    'MinMaxNormalizer',
    'normalize_features',
    'rank_normalize',
    'zscore_normalize'
]
