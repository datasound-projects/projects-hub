"""
Symbolic regression module for alpha discovery.

This module implements the PySR-based symbolic regression with grammar constraints.
"""

from .pysr_config import (
    PySRConfig,
    get_default_config,
    create_pysr_config
)
from .regression import (
    SymbolicRegressor,
    run_symbolic_regression,
    discover_alphas
)
from .expressions import (
    AlphaExpression,
    ExpressionAnalyzer,
    simplify_expression,
    validate_expression
)

__all__ = [
    # Configuration
    'PySRConfig', 'get_default_config', 'create_pysr_config',
    # Regression
    'SymbolicRegressor', 'run_symbolic_regression', 'discover_alphas',
    # Expressions
    'AlphaExpression', 'ExpressionAnalyzer', 'simplify_expression', 'validate_expression'
]
