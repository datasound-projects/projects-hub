"""
Threshold-based policies.

This module implements policies that use thresholds to determine positions.
"""

import pandas as pd
import numpy as np
from typing import Optional
from dataclasses import dataclass
import logging

from .base import Policy, PolicyConfig, ParameterizedPolicy

logger = logging.getLogger(__name__)


@dataclass
class ThresholdConfig(PolicyConfig):
    """Configuration for threshold policy."""
    # Threshold for going long
    long_threshold: float = 0.0
    
    # Threshold for going short
    short_threshold: float = 0.0
    
    # Position size when long
    long_size: float = 1.0
    
    # Position size when short
    short_size: float = 1.0


class ThresholdPolicy(ParameterizedPolicy):
    """
    Threshold-based policy.
    
    Goes long when alpha > long_threshold, short when alpha < short_threshold,
    and flat otherwise.
    """
    
    def __init__(self, 
                 config: Optional[ThresholdConfig] = None):
        """
        Initialize the threshold policy.
        
        Args:
            config: Threshold configuration
        """
        if config is None:
            config = ThresholdConfig(
                name="threshold",
                description="Threshold-based policy",
                is_parameterized=True,
                parameters={
                    'long_threshold': 0.0,
                    'short_threshold': 0.0,
                    'long_size': 1.0,
                    'short_size': 1.0
                }
            )
        super().__init__(config)
    
    def compute_weights(self, 
                        alpha: pd.Series) -> pd.Series:
        """
        Compute portfolio weights.
        
        Args:
            alpha: Series of alpha scores
            
        Returns:
            Series of portfolio weights
        """
        long_threshold = self.config.parameters.get('long_threshold', 0.0)
        short_threshold = self.config.parameters.get('short_threshold', 0.0)
        long_size = self.config.parameters.get('long_size', 1.0)
        short_size = self.config.parameters.get('short_size', 1.0)
        
        weights = pd.Series(0.0, index=alpha.index)
        
        # Long positions
        long_mask = alpha > long_threshold
        weights[long_mask] = long_size
        
        # Short positions
        short_mask = alpha < short_threshold
        weights[short_mask] = -short_size
        
        # Normalize to sum to 1 (or -1 for long-short)
        total_long = weights[weights > 0].sum()
        total_short = weights[weights < 0].sum()
        
        if total_long > 0:
            weights[weights > 0] = weights[weights > 0] / total_long
        
        if total_short < 0:
            weights[weights < 0] = weights[weights < 0] / abs(total_short)
        
        # Balance long and short
        if total_long > 0 and total_short < 0:
            weights = weights / (total_long + abs(total_short))
        
        return weights
    
    def get_parameter_grid(self) -> List[Dict[str, float]]:
        """
        Get a grid of parameter values for optimization.
        
        Returns:
            List of parameter dictionaries
        """
        thresholds = [-1.0, -0.5, 0.0, 0.5, 1.0]
        sizes = [0.5, 1.0, 1.5, 2.0]
        
        grid = []
        for long_thresh in thresholds:
            for short_thresh in thresholds:
                for long_size in sizes:
                    for short_size in sizes:
                        if long_thresh >= short_thresh:  # Ensure long threshold > short threshold
                            grid.append({
                                'long_threshold': long_thresh,
                                'short_threshold': short_thresh,
                                'long_size': long_size,
                                'short_size': short_size
                            })
        
        return grid


class ZScorePolicy(ParameterizedPolicy):
    """
    Z-score proportional policy.
    
    Sizes positions proportionally to the cross-sectional z-score of alpha,
    clipped at a specified number of standard deviations.
    """
    
    def __init__(self, 
                 config: Optional[PolicyConfig] = None):
        """
        Initialize the z-score policy.
        
        Args:
            config: Policy configuration
        """
        if config is None:
            config = PolicyConfig(
                name="zscore",
                description="Z-score proportional policy",
                is_parameterized=True,
                parameters={
                    'clip_value': 2.0,
                    'normalize': True
                }
            )
        super().__init__(config)
    
    def compute_weights(self, 
                        alpha: pd.Series) -> pd.Series:
        """
        Compute portfolio weights.
        
        Args:
            alpha: Series of alpha scores
            
        Returns:
            Series of portfolio weights
        """
        clip_value = self.config.parameters.get('clip_value', 2.0)
        normalize = self.config.parameters.get('normalize', True)
        
        # Compute z-scores
        z_scores = (alpha - alpha.mean()) / alpha.std()
        
        # Clip at clip_value
        z_scores = z_scores.clip(-clip_value, clip_value)
        
        # Create weights
        weights = z_scores
        
        # Normalize if requested
        if normalize and weights.abs().sum() > 0:
            weights = weights / weights.abs().sum()
        
        return weights
    
    def get_parameter_grid(self) -> List[Dict[str, float]]:
        """
        Get a grid of parameter values for optimization.
        
        Returns:
            List of parameter dictionaries
        """
        clip_values = [1.0, 1.5, 2.0, 2.5, 3.0]
        
        grid = []
        for clip in clip_values:
            for normalize in [True, False]:
                grid.append({
                    'clip_value': clip,
                    'normalize': normalize
                })
        
        return grid


class SignPolicy(Policy):
    """
    Sign policy.
    
    Goes long when alpha > 0 and short when alpha < 0 with equal position sizes.
    """
    
    def __init__(self, 
                 config: Optional[PolicyConfig] = None):
        """
        Initialize the sign policy.
        
        Args:
            config: Policy configuration
        """
        if config is None:
            config = PolicyConfig(
                name="sign",
                description="Sign-based policy: long when alpha > 0, short when alpha < 0"
            )
        super().__init__(config)
    
    def compute_weights(self, 
                        alpha: pd.Series) -> pd.Series:
        """
        Compute portfolio weights.
        
        Args:
            alpha: Series of alpha scores
            
        Returns:
            Series of portfolio weights
        """
        weights = pd.Series(0.0, index=alpha.index)
        
        # Long positions
        long_mask = alpha > 0
        weights[long_mask] = 1.0
        
        # Short positions
        short_mask = alpha < 0
        weights[short_mask] = -1.0
        
        # Normalize
        total_long = weights[weights > 0].sum()
        total_short = weights[weights < 0].sum()
        
        if total_long > 0:
            weights[weights > 0] = weights[weights > 0] / total_long
        
        if total_short < 0:
            weights[weights < 0] = weights[weights < 0] / abs(total_short)
        
        # Balance long and short
        if total_long > 0 and total_short < 0:
            weights = weights / (total_long + abs(total_short))
        
        return weights
