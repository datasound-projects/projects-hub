"""
Quintile long-short policy.

This module implements the quintile long-short policy, which is the
primary policy used for validation in the methodology.
"""

import pandas as pd
import numpy as np
from typing import Optional, List
from dataclasses import dataclass
import logging

from .base import Policy, PolicyConfig

logger = logging.getLogger(__name__)


@dataclass
class QuintileConfig(PolicyConfig):
    """Configuration for quintile long-short policy."""
    # Number of quintiles
    n_quintiles: int = 5
    
    # Whether to use equal weighting within quintiles
    equal_weight: bool = True
    
    # Whether to normalize weights to sum to 1
    normalize: bool = True


class QuintileLongShortPolicy(Policy):
    """
    Quintile long-short policy.
    
    Goes long the top quintile of assets by alpha and short the bottom quintile,
    with equal weights within each quintile.
    
    This is the primary policy used for validation in the methodology.
    """
    
    def __init__(self, 
                 config: Optional[QuintileConfig] = None):
        """
        Initialize the quintile long-short policy.
        
        Args:
            config: Quintile configuration
        """
        if config is None:
            config = QuintileConfig(
                name="quintile_long_short",
                description="Long top quintile, short bottom quintile"
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
        n = len(alpha)
        n_quintiles = self.config.parameters.get('n_quintiles', 5)
        
        # Rank the alpha scores
        ranked = alpha.rank(method="average")
        
        # Determine quintile boundaries
        n_per_quintile = n // n_quintiles
        
        # Initialize weights
        weights = pd.Series(0.0, index=alpha.index)
        
        # Long the top quintile
        top_quintile = ranked >= n - n_per_quintile
        if top_quintile.any():
            if self.config.parameters.get('equal_weight', True):
                weights[top_quintile] = 1.0 / n_per_quintile
            else:
                # Weight by alpha within quintile
                top_alpha = alpha[top_quintile]
                top_alpha_normalized = top_alpha - top_alpha.min()
                top_alpha_normalized = top_alpha_normalized / top_alpha_normalized.sum()
                weights[top_quintile] = top_alpha_normalized
        
        # Short the bottom quintile
        bottom_quintile = ranked <= n_per_quintile
        if bottom_quintile.any():
            if self.config.parameters.get('equal_weight', True):
                weights[bottom_quintile] -= 1.0 / n_per_quintile
            else:
                # Weight by alpha within quintile (more negative = more short)
                bottom_alpha = alpha[bottom_quintile]
                bottom_alpha_normalized = -bottom_alpha + bottom_alpha.max()
                bottom_alpha_normalized = bottom_alpha_normalized / bottom_alpha_normalized.sum()
                weights[bottom_quintile] -= bottom_alpha_normalized
        
        # Normalize if requested
        if self.config.parameters.get('normalize', True) and weights.abs().sum() > 0:
            weights = weights / weights.abs().sum()
        
        return weights


class DecileLongShortPolicy(QuintileLongShortPolicy):
    """
    Decile long-short policy.
    
    Similar to quintile but uses 10 groups (deciles).
    """
    
    def __init__(self, 
                 config: Optional[QuintileConfig] = None):
        if config is None:
            config = QuintileConfig(
                name="decile_long_short",
                description="Long top decile, short bottom decile",
                n_quintiles=10
            )
        super().__init__(config)


class TertileLongShortPolicy(QuintileLongShortPolicy):
    """
    Tertile long-short policy.
    
    Uses 3 groups (tertiles).
    """
    
    def __init__(self, 
                 config: Optional[QuintileConfig] = None):
        if config is None:
            config = QuintileConfig(
                name="tertile_long_short",
                description="Long top tertile, short bottom tertile",
                n_quintiles=3
            )
        super().__init__(config)
