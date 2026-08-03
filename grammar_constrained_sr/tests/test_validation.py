"""
Tests for validation module.
"""

import pandas as pd
import numpy as np
import pytest
from src.validation.metrics import compute_ic, compute_ic_stability, compute_sharpe_ratio


def test_ic_computation():
    """Test Information Coefficient computation."""
    # Create test data with perfect correlation
    np.random.seed(42)
    n = 100
    
    # Perfect positive correlation
    alpha = pd.Series(np.random.normal(0, 1, n))
    target = alpha * 2 + 0.1  # Linear relationship
    
    ic = compute_ic(alpha, target)
    
    # Should be close to 1 (perfect correlation)
    assert ic > 0.9
    
    # Perfect negative correlation
    target_neg = -alpha
    ic_neg = compute_ic(alpha, target_neg)
    
    # Should be close to -1
    assert ic_neg < -0.9
    
    # No correlation
    target_random = pd.Series(np.random.normal(0, 1, n))
    ic_random = compute_ic(alpha, target_random)
    
    # Should be close to 0
    assert abs(ic_random) < 0.2


def test_sharpe_ratio():
    """Test Sharpe ratio computation."""
    # Create test data
    np.random.seed(42)
    n = 100
    
    # Create panel data
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    assets = [f"asset_{i}" for i in range(10)]
    
    index = pd.MultiIndex.from_product([dates, assets], names=["date", "symbol"])
    
    # Create alpha and target data
    alpha = pd.Series(np.random.normal(0, 1, 100), index=index)
    target = pd.Series(np.random.normal(0.01, 0.02, 100), index=index)
    
    # Compute Sharpe ratio
    sharpe = compute_sharpe_ratio(alpha, target, n_quintiles=2)
    
    # Should be a finite number
    assert np.isfinite(sharpe)
    
    # Test with perfect momentum signal
    # Top assets have positive returns, bottom have negative
    alpha_perfect = pd.Series(0.0, index=index)
    for date in dates:
        date_alpha = alpha_perfect.loc[date]
        date_alpha.iloc[:5] = -1  # Bottom half
        date_alpha.iloc[5:] = 1   # Top half
    
    target_perfect = pd.Series(0.0, index=index)
    for date in dates:
        date_target = target_perfect.loc[date]
        date_target.iloc[:5] = -0.02  # Negative returns
        date_target.iloc[5:] = 0.02   # Positive returns
    
    sharpe_perfect = compute_sharpe_ratio(alpha_perfect, target_perfect, n_quintiles=2)
    
    # Should be positive (good signal)
    assert sharpe_perfect > 0


if __name__ == "__main__":
    test_ic_computation()
    test_sharpe_ratio()
    print("All validation tests passed!")
