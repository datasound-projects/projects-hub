"""
Examples for Grammar-Constrained Symbolic Regression for Alpha Discovery

This module contains example scripts demonstrating various aspects of the
alpha discovery pipeline.
"""

from .basic_example import run_basic_example
from .synthetic_recovery import run_synthetic_recovery_test
from .full_pipeline import run_full_pipeline_example

__all__ = [
    'run_basic_example',
    'run_synthetic_recovery_test',
    'run_full_pipeline_example'
]
