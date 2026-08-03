"""
PySR configuration for grammar-constrained symbolic regression.

This module provides configuration for PySR that enforces the grammar
constraints described in the methodology.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class PySRConfig:
    """
    Configuration for PySR symbolic regression.
    
    Enforces grammar constraints by limiting operators, setting complexity
    penalties, and controlling the search space.
    """
    
    # ============================================================================
    # Search Space Configuration
    # ============================================================================
    
    # Binary operators: restricted to basic algebraic operations
    binary_operators: List[str] = field(default_factory=lambda: ["+", "-", "*", "/"])
    
    # Unary operators: restricted to abs and neg
    unary_operators: List[str] = field(default_factory=lambda: ["abs", "neg"])
    
    # Maximum expression size (number of nodes)
    max_size: int = 18
    
    # Maximum depth of expression tree
    max_depth: Optional[int] = 5
    
    # ============================================================================
    # Complexity and Parsimony
    # ============================================================================
    
    # Parsimony coefficient (higher = more pressure for simpler expressions)
    parsimony: float = 0.012
    
    # Complexity penalties for specific operators
    complexity_penalties: Dict[str, float] = field(default_factory=lambda: {
        "/": 2.0,  # Division is more expensive
        "+": 1.0,
        "-": 1.0,
        "*": 1.0,
        "abs": 1.0,
        "neg": 1.0
    })
    
    # ============================================================================
    # Optimization Parameters
    # ============================================================================
    
    # Number of populations
    n_populations: int = 30
    
    # Population size
    population_size: int = 50
    
    # Maximum iterations
    max_iterations: int = 100
    
    # Maximum time in seconds
    max_time: int = 900  # 15 minutes
    
    # ============================================================================
    # Loss Function
    # ============================================================================
    
    # Loss function: 'L1' (MAE) or 'L2' (MSE)
    loss_function: str = "L1"
    
    # ============================================================================
    # Numerical Stability
    # ============================================================================
    
    # Whether to use deterministic mode for reproducibility
    deterministic: bool = True
    
    # Random seed for deterministic mode
    random_seed: Optional[int] = 42
    
    # ============================================================================
    # Output Control
    # ============================================================================
    
    # Number of top expressions to return
    n_top_expressions: int = 100
    
    # Whether to include complexity in the output
    include_complexity: bool = True
    
    # Whether to include loss in the output
    include_loss: bool = True
    
    # ============================================================================
    # Advanced Constraints
    # ============================================================================
    
    # Whether to allow division by zero (should be False)
    allow_division_by_zero: bool = False
    
    # Whether to allow nested operations (e.g., abs(abs(x)))
    allow_nested_operations: bool = True
    
    # Maximum nesting depth
    max_nesting_depth: int = 2
    
    # ============================================================================
    # Feature Selection
    # ============================================================================
    
    # Whether to use feature selection
    use_feature_selection: bool = False
    
    # Number of features to select
    n_features_to_select: Optional[int] = None
    
    # ============================================================================
    # Early Stopping
    # ============================================================================
    
    # Whether to use early stopping
    early_stopping: bool = True
    
    # Early stopping patience (number of iterations without improvement)
    early_stopping_patience: int = 10


def get_default_config() -> PySRConfig:
    """
    Get the default PySR configuration for alpha discovery.
    
    Returns:
        Default PySRConfig with grammar constraints
    """
    return PySRConfig()


def create_pysr_config(
    max_size: int = 18,
    parsimony: float = 0.012,
    n_populations: int = 30,
    population_size: int = 50,
    max_iterations: int = 100,
    max_time: int = 900,
    loss_function: str = "L1",
    deterministic: bool = True,
    random_seed: Optional[int] = 42
) -> PySRConfig:
    """
    Create a custom PySR configuration.
    
    Args:
        max_size: Maximum expression size
        parsimony: Parsimony coefficient
        n_populations: Number of populations
        population_size: Population size
        max_iterations: Maximum iterations
        max_time: Maximum time in seconds
        loss_function: Loss function ('L1' or 'L2')
        deterministic: Whether to use deterministic mode
        random_seed: Random seed
        
    Returns:
        Custom PySRConfig
    """
    return PySRConfig(
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["abs", "neg"],
        max_size=max_size,
        max_depth=5,
        parsimony=parsimony,
        complexity_penalties={"/": 2.0, "+": 1.0, "-": 1.0, "*": 1.0, "abs": 1.0, "neg": 1.0},
        n_populations=n_populations,
        population_size=population_size,
        max_iterations=max_iterations,
        max_time=max_time,
        loss_function=loss_function,
        deterministic=deterministic,
        random_seed=random_seed,
        n_top_expressions=100,
        include_complexity=True,
        include_loss=True,
        allow_division_by_zero=False,
        allow_nested_operations=True,
        max_nesting_depth=2,
        early_stopping=True,
        early_stopping_patience=10
    )


def config_to_dict(config: PySRConfig) -> Dict[str, Any]:
    """
    Convert PySRConfig to a dictionary for PySR.
    
    Args:
        config: PySRConfig object
        
    Returns:
        Dictionary with PySR parameters
    """
    return {
        "binary_operators": config.binary_operators,
        "unary_operators": config.unary_operators,
        "maxsize": config.max_size,
        "maxdepth": config.max_depth,
        "parsimony": config.parsimony,
        "npop": config.n_populations,
        "niterations": config.max_iterations,
        "maxtime": config.max_time,
        "loss_function": config.loss_function,
        "deterministic": config.deterministic,
        "random_state": config.random_seed,
        "early_stop_condition": "loss" if config.early_stopping else None,
        "early_stop_patience": config.early_stopping_patience,
        "complexity_of_operators": config.complexity_penalties,
    }
