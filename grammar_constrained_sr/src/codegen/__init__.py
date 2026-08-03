"""
Code generation module for alpha expressions.

This module implements deterministic code generation via SymPy for deploying
discovered alpha expressions to backtesting infrastructure.
"""

from .sympy_utils import (
    SymPyUtils,
    expression_to_sympy,
    sympy_to_python,
    sympy_to_numpy,
    sympy_to_c
)
from .generator import (
    CodeGenerator,
    generate_alpha_code,
    generate_portfolio_code,
    generate_backtest_code
)

__all__ = [
    # SymPy utilities
    'SymPyUtils', 'expression_to_sympy', 'sympy_to_python',
    'sympy_to_numpy', 'sympy_to_c',
    # Generator
    'CodeGenerator', 'generate_alpha_code', 'generate_portfolio_code',
    'generate_backtest_code'
]
