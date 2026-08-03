"""
Cross-sectional normalization for Layer 2 of the architecture.

This module provides functionality to normalize features cross-sectionally,
making them comparable across assets. This is essential for creating
cross-sectional alpha signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class NormalizationConfig:
    """Configuration for cross-sectional normalization."""
    # Type of normalization: 'rank', 'zscore', 'minmax', or 'none'
    method: str = 'rank'
    
    # For rank normalization: percentile range (0-100)
    percentile_range: Tuple[float, float] = (0.0, 100.0)
    
    # For z-score: clip outliers beyond this many standard deviations
    zscore_clip: Optional[float] = 3.0
    
    # For min-max: clip to this percentile range before scaling
    minmax_clip_percentile: Optional[Tuple[float, float]] = (1.0, 99.0)
    
    # Whether to handle NaN values
    handle_nan: bool = True
    
    # How to handle NaN: 'drop', 'fill', or 'ignore'
    nan_handling: str = 'ignore'


class CrossSectionalNormalizer:
    """
    Base class for cross-sectional normalization.
    
    Provides common functionality for normalizing features across assets
    at each point in time.
    """
    
    def __init__(self, 
                 config: Optional[NormalizationConfig] = None):
        """
        Initialize the normalizer.
        
        Args:
            config: Normalization configuration
        """
        self.config = config or NormalizationConfig()
    
    def normalize(self, 
                 features: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features cross-sectionally.
        
        Args:
            features: Input feature DataFrame with MultiIndex (symbol, date)
            
        Returns:
            Normalized feature DataFrame
        """
        # This should be implemented by subclasses
        pass
    
    def _handle_nan(self, 
                   data: pd.DataFrame) -> pd.DataFrame:
        """Handle NaN values according to configuration."""
        if self.config.nan_handling == 'drop':
            return data.dropna()
        elif self.config.nan_handling == 'fill':
            return data.fillna(method='ffill').fillna(method='bfill')
        else:  # 'ignore'
            return data


class PercentileRankNormalizer(CrossSectionalNormalizer):
    """
    Percentile rank normalization.
    
    Normalizes features to [0, 1] range based on their percentile rank
    across assets at each point in time.
    
    Formula: rank(x) = (number of values <= x) / (total number of values)
    """
    
    def __init__(self, 
                 config: Optional[NormalizationConfig] = None):
        config = config or NormalizationConfig(method='rank')
        super().__init__(config)
    
    def normalize(self, 
                 features: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features using percentile rank.
        
        Args:
            features: Input feature DataFrame
            
        Returns:
            DataFrame with percentile rank normalized features
        """
        logger.info("Applying percentile rank normalization...")
        
        # Handle NaN values
        features = self._handle_nan(features)
        
        # Apply percentile rank normalization to each feature
        normalized = features.copy()
        
        for col in features.columns:
            # Group by date and compute rank
            normalized[col] = features.groupby(level='date')[col].rank(
                pct=True,
                method='average'
            )
            
            # Scale to the desired percentile range
            if self.config.percentile_range != (0.0, 100.0):
                min_pct, max_pct = self.config.percentile_range
                normalized[col] = min_pct + (max_pct - min_pct) * normalized[col]
        
        return normalized


class ZScoreNormalizer(CrossSectionalNormalizer):
    """
    Z-score normalization.
    
    Normalizes features to have mean 0 and standard deviation 1 across
    assets at each point in time.
    
    Formula: z(x) = (x - mean(x)) / std(x)
    """
    
    def __init__(self, 
                 config: Optional[NormalizationConfig] = None):
        config = config or NormalizationConfig(method='zscore')
        super().__init__(config)
    
    def normalize(self, 
                 features: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features using z-score.
        
        Args:
            features: Input feature DataFrame
            
        Returns:
            DataFrame with z-score normalized features
        """
        logger.info("Applying z-score normalization...")
        
        # Handle NaN values
        features = self._handle_nan(features)
        
        # Apply z-score normalization to each feature
        normalized = features.copy()
        
        for col in features.columns:
            # Group by date and compute z-score
            grouped = features.groupby(level='date')[col]
            mean = grouped.transform('mean')
            std = grouped.transform('std')
            
            normalized[col] = (features[col] - mean) / std.replace(0, np.nan)
            
            # Clip outliers if specified
            if self.config.zscore_clip:
                normalized[col] = normalized[col].clip(
                    -self.config.zscore_clip,
                    self.config.zscore_clip
                )
        
        return normalized


class MinMaxNormalizer(CrossSectionalNormalizer):
    """
    Min-max normalization.
    
    Normalizes features to [0, 1] range based on min and max values
    across assets at each point in time.
    
    Formula: minmax(x) = (x - min(x)) / (max(x) - min(x))
    """
    
    def __init__(self, 
                 config: Optional[NormalizationConfig] = None):
        config = config or NormalizationConfig(method='minmax')
        super().__init__(config)
    
    def normalize(self, 
                 features: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features using min-max scaling.
        
        Args:
            features: Input feature DataFrame
            
        Returns:
            DataFrame with min-max normalized features
        """
        logger.info("Applying min-max normalization...")
        
        # Handle NaN values
        features = self._handle_nan(features)
        
        # Apply min-max normalization to each feature
        normalized = features.copy()
        
        for col in features.columns:
            # Group by date
            grouped = features.groupby(level='date')[col]
            min_val = grouped.transform('min')
            max_val = grouped.transform('max')
            
            # Avoid division by zero
            range_val = max_val - min_val
            range_val = range_val.replace(0, np.nan)
            
            normalized[col] = (features[col] - min_val) / range_val
            
            # Clip to [0, 1] range
            normalized[col] = normalized[col].clip(0, 1)
        
        return normalized


class CombinedNormalizer:
    """
    Combined normalizer that applies multiple normalization methods.
    
    Can create both raw and normalized versions of features for PySR
    to use in its search.
    """
    
    def __init__(self, 
                 config: Optional[NormalizationConfig] = None,
                 include_raw: bool = True,
                 include_rank: bool = True,
                 include_zscore: bool = True):
        """
        Initialize the combined normalizer.
        
        Args:
            config: Base normalization configuration
            include_raw: Whether to include raw (unnormalized) features
            include_rank: Whether to include percentile rank normalized features
            include_zscore: Whether to include z-score normalized features
        """
        self.config = config or NormalizationConfig()
        self.include_raw = include_raw
        self.include_rank = include_rank
        self.include_zscore = include_zscore
        
        # Create individual normalizers
        self.rank_normalizer = PercentileRankNormalizer(self.config)
        self.zscore_normalizer = ZScoreNormalizer(self.config)
    
    def normalize(self, 
                 features: pd.DataFrame) -> pd.DataFrame:
        """
        Apply multiple normalization methods.
        
        Args:
            features: Input feature DataFrame
            
        Returns:
            DataFrame with all normalized versions of features
        """
        logger.info("Applying combined normalization...")
        
        # Start with raw features
        all_features = []
        
        if self.include_raw:
            all_features.append(features.add_prefix('raw_'))
        
        # Add rank-normalized features
        if self.include_rank:
            rank_features = self.rank_normalizer.normalize(features)
            all_features.append(rank_features.add_prefix('rank_'))
        
        # Add z-score normalized features
        if self.include_zscore:
            zscore_features = self.zscore_normalizer.normalize(features)
            all_features.append(zscore_features.add_prefix('zscore_'))
        
        # Combine all features
        result = pd.concat(all_features, axis=1)
        
        logger.info(f"Created {len(result.columns)} features (raw: {self.include_raw}, rank: {self.include_rank}, zscore: {self.include_zscore})")
        
        return result


def normalize_features(features: pd.DataFrame,
                      method: str = 'rank',
                      **kwargs) -> pd.DataFrame:
    """
    Convenience function to normalize features.
    
    Args:
        features: Input feature DataFrame
        method: Normalization method ('rank', 'zscore', 'minmax')
        **kwargs: Additional arguments for the normalizer
        
    Returns:
        Normalized feature DataFrame
    """
    config = NormalizationConfig(method=method, **kwargs)
    
    if method == 'rank':
        normalizer = PercentileRankNormalizer(config)
    elif method == 'zscore':
        normalizer = ZScoreNormalizer(config)
    elif method == 'minmax':
        normalizer = MinMaxNormalizer(config)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalizer.normalize(features)


def rank_normalize(features: pd.DataFrame,
                  percentile_range: Tuple[float, float] = (0.0, 100.0)) -> pd.DataFrame:
    """
    Convenience function for percentile rank normalization.
    
    Args:
        features: Input feature DataFrame
        percentile_range: Range for percentile scaling
        
    Returns:
        Rank-normalized feature DataFrame
    """
    config = NormalizationConfig(
        method='rank',
        percentile_range=percentile_range
    )
    normalizer = PercentileRankNormalizer(config)
    return normalizer.normalize(features)


def zscore_normalize(features: pd.DataFrame,
                    clip: Optional[float] = 3.0) -> pd.DataFrame:
    """
    Convenience function for z-score normalization.
    
    Args:
        features: Input feature DataFrame
        clip: Number of standard deviations to clip outliers
        
    Returns:
        Z-score normalized feature DataFrame
    """
    config = NormalizationConfig(
        method='zscore',
        zscore_clip=clip
    )
    normalizer = ZScoreNormalizer(config)
    return normalizer.normalize(features)
