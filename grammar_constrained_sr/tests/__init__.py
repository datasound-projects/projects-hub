"""
Test suite for Grammar-Constrained Symbolic Regression for Alpha Discovery
"""

from .test_features import test_volatility_primitives, test_autocorrelation_primitives
from .test_normalization import test_rank_normalization, test_zscore_normalization
from .test_symbolic import test_expression_validation, test_sympy_conversion
from .test_validation import test_ic_computation, test_sharpe_ratio
from .test_codegen import test_code_generation

__all__ = [
    'test_volatility_primitives', 'test_autocorrelation_primitives',
    'test_rank_normalization', 'test_zscore_normalization',
    'test_expression_validation', 'test_sympy_conversion',
    'test_ic_computation', 'test_sharpe_ratio',
    'test_code_generation'
]
