"""
Base classes for policy functions.

This module defines the base classes and interfaces for all policy functions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class PolicyConfig:
    """Base configuration for a policy function."""
    # Name of the policy
    name: str = "base_policy"
    
    # Description of the policy
    description: str = ""
    
    # Whether the policy is parameterized
    is_parameterized: bool = False
    
    # Parameters for the policy
    parameters: Dict[str, Any] = field(default_factory=dict)


class Policy(ABC):
    """
    Abstract base class for all policy functions.
    
    A policy function takes alpha scores as input and produces trading
    actions (position sizes) as output.
    """
    
    def __init__(self, 
                 config: Optional[PolicyConfig] = None):
        """
        Initialize the policy.
        
        Args:
            config: Policy configuration
        """
        self.config = config or PolicyConfig()
    
    @abstractmethod
    def compute_weights(self, 
                        alpha: pd.Series) -> pd.Series:
        """
        Compute portfolio weights from alpha scores.
        
        Args:
            alpha: Series of alpha scores for each asset
            
        Returns:
            Series of portfolio weights for each asset
        """
        pass
    
    def apply(self, 
             alpha: pd.Series,
             capital: float = 1.0) -> pd.Series:
        """
        Apply the policy to alpha scores.
        
        Args:
            alpha: Series of alpha scores
            capital: Total capital to allocate
            
        Returns:
            Series of position sizes (in capital units)
        """
        weights = self.compute_weights(alpha)
        
        # Normalize weights to sum to 1 (or -1 for long-short)
        if weights.sum() != 0:
            weights = weights / abs(weights.sum())
        
        # Scale by capital
        positions = weights * capital
        
        return positions
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get the current parameters of the policy."""
        return self.config.parameters if self.config else {}
    
    def set_parameters(self, **kwargs) -> None:
        """Set parameters for the policy."""
        if not self.config:
            self.config = PolicyConfig()
        self.config.parameters.update(kwargs)


class ParameterizedPolicy(Policy):
    """
    Base class for parameterized policies.
    
    Policies that have tunable parameters should inherit from this class.
    """
    
    def __init__(self, 
                 config: Optional[PolicyConfig] = None):
        if config is None:
            config = PolicyConfig(is_parameterized=True)
        super().__init__(config)
    
    def get_parameter_grid(self) -> List[Dict[str, Any]]:
        """
        Get a grid of parameter values for optimization.
        
        Returns:
            List of parameter dictionaries
        """
        # This should be implemented by subclasses
        return [self.get_parameters()]
