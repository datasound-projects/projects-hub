"""
Pipeline module for alpha discovery.

This module implements the complete pipeline for grammar-constrained
symbolic regression for alpha discovery.
"""

from .main import AlphaDiscoveryPipeline
from .workflow import WorkflowManager

__all__ = [
    'AlphaDiscoveryPipeline',
    'WorkflowManager'
]
