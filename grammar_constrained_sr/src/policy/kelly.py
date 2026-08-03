"""
Kelly criterion policy.

This module implements the Kelly criterion for position sizing,
which maximizes the expected growth rate of capital.
"""

import pandas as pd
import numpy as np
from typing import Optional
from dataclasses import dataclass
import logging

from .base import Policy, PolicyConfig, ParameterizedPolicy

logger = logging.getLogger(__name__)


@dataclass
class KellyConfig(PolicyConfig):
    """Configuration for Kelly criterion policy."""
    # Fraction of capital to risk (0.0 to 1.0)
    fraction: float = 1.0
    
    # Minimum position size
    min_position: float = 0.0
    
    # Maximum position size
    max_position: float = 1.0


class KellyCriterionPolicy(ParameterizedPolicy):
    """
    Kelly criterion policy.
    
    Sizes positions according to the Kelly criterion: f* = (mu / sigma^2),
    where mu is the expected return and sigma^2 is the variance.
    
    This policy maximizes the expected growth rate of capital.
    """
    
    def __init__(self, 
                 config: Optional[KellyConfig] = None):
        """
        Initialize the Kelly criterion policy.
        
        Args:
            config: Kelly configuration
        """
        if config is None:
            config = KellyConfig(
                name="kelly",
                description="Kelly criterion position sizing",
                is_parameterized=True,
                parameters={
                    'fraction': 1.0,
                    'min_position': 0.0,
                    'max_position': 1.0
                }
            )
        super().__init__(config)
    
    def compute_weights(self, 
                        alpha: pd.Series,
                        returns: Optional[pd.Series] = None) -> pd.Series:
        """
        Compute portfolio weights using Kelly criterion.
        
        Args:
            alpha: Series of alpha scores (used as expected returns)
            returns: Optional historical returns for variance estimation
            
        Returns:
            Series of portfolio weights
        """
        fraction = self.config.parameters.get('fraction', 1.0)
        min_position = self.config.parameters.get('min_position', 0.0)
        max_position = self.config.parameters.get('max_position', 1.0)
        
        # If returns are provided, use them for variance estimation
        if returns is not None:
            # Estimate variance from historical returns
            variance = returns.var()
            if variance <= 0:
                variance = 1.0
        else:
            # Use alpha variance as proxy
            variance = alpha.var()
            if variance <= 0:
                variance = 1.0
        
        # Kelly formula: f* = (mu / sigma^2)
        # Here, alpha serves as mu (expected return)
        kelly_weights = alpha / variance
        
        # Apply fraction
        kelly_weights = kelly_weights * fraction
        
        # Clip to min/max
        kelly_weights = kelly_weights.clip(min_position, max_position)
        
        # Normalize to sum to 1 (or -1 for long-short)
        total_long = kelly_weights[kelly_weights > 0].sum()
        total_short = kelly_weights[kelly_weights < 0].sum()
        
        if total_long > 0:
            kelly_weights[kelly_weights > 0] = kelly_weights[kelly_weights > 0] / total_long
        
        if total_short < 0:
            kelly_weights[kelly_weights < 0] = kelly_weights[kelly_weights < 0] / abs(total_short)
        
        # Balance long and short
        if total_long > 0 and total_short < 0:
            kelly_weights = kelly_weights / (total_long + abs(total_short))
        
        return kelly_weights
    
    def get_parameter_grid(self) -> list:
        """
        Get a grid of parameter values for optimization.
        
        Returns:
            List of parameter dictionaries
        """
        fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
        min_positions = [0.0, 0.1, 0.2]
        max_positions = [0.5, 1.0, 1.5, 2.0]
        
        grid = []
        for fraction in fractions:
            for min_pos in min_positions:
                for max_pos in max_positions:
                    if min_pos < max_pos:
                        grid.append({
                            'fraction': fraction,
                            'min_position': min_pos,
                            'max_position': max_pos
                        })
        
        return grid


class HalfKellyPolicy(KellyCriterionPolicy):
    """
    Half-Kelly policy.
    
    Uses half the Kelly criterion position size for more conservative sizing.
    """
    
    def __init__(self, 
                 config: Optional[KellyConfig] = None):
        if config is None:
            config = KellyConfig(
                name="half_kelly",
                description="Half-Kelly criterion position sizing",
                fraction=0.5
            )
        super().__init__(config)


class QuarterKellyPolicy(KellyCriterionPolicy):
    """
    Quarter-Kelly policy.
    
    Uses one-quarter the Kelly criterion position size for very conservative sizing.
    """
    
    def __init__(self, 
                 config: Optional[KellyConfig] = None):
        if config is None:
            config = KellyConfig(
                name="quarter_kelly",
                description="Quarter-Kelly criterion position sizing",
                fraction=0.25
            )
        super().__init__(config)
