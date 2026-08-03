"""
Grammar-Constrained Symbolic Regression for Systematic Alpha Discovery

A methodology for discovering quantitative trading signals (alphas) using symbolic regression
constrained by a grammar derived from empirical regularities of financial time series.
"""

__version__ = "0.1.0"
__author__ = "Patryk Kozak"
__email__ = "patryk.kozak@datasound.projects"

from .data import loader, panel, synthetic
from .features import factory
from .normalization import cross_sectional
from .symbolic import pysr_config, regression, expressions
from .validation import temporal, statistical, metrics
from .policy import base, quintile, threshold, kelly, optimizer
from .codegen import sympy_utils, generator
from .pipeline import main, workflow

__all__ = [
    # Data modules
    'loader', 'panel', 'synthetic',
    # Features
    'factory',
    # Normalization
    'cross_sectional',
    # Symbolic regression
    'pysr_config', 'regression', 'expressions',
    # Validation
    'temporal', 'statistical', 'metrics',
    # Policy
    'base', 'quintile', 'threshold', 'kelly', 'optimizer',
    # Code generation
    'sympy_utils', 'generator',
    # Pipeline
    'main', 'workflow'
]
