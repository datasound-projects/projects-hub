"""
Validation module for alpha discovery.

This module implements the validation protocol including:
- Temporal splits
- Statistical filtering
- Performance metrics
"""

from .temporal import (
    TemporalSplitter,
    create_temporal_splits,
    train_test_split_temporal
)
from .statistical import (
    StatisticalFilter,
    FilterConfig,
    filter_candidates,
    compute_ic,
    compute_sharpe_ratio,
    compute_turnover
)
from .metrics import (
    InformationCoefficient,
    PerformanceMetrics,
    compute_all_metrics,
    compute_ic_stability
)

__all__ = [
    # Temporal
    'TemporalSplitter', 'create_temporal_splits', 'train_test_split_temporal',
    # Statistical
    'StatisticalFilter', 'FilterConfig', 'filter_candidates',
    'compute_ic', 'compute_sharpe_ratio', 'compute_turnover',
    # Metrics
    'InformationCoefficient', 'PerformanceMetrics',
    'compute_all_metrics', 'compute_ic_stability'
]
