"""
Tests for normalization module.
"""

import pandas as pd
import numpy as np
import pytest
from src.normalization import rank_normalize, zscore_normalize, CombinedNormalizer


def test_rank_normalization():
    """Test percentile rank normalization."""
    # Create test data
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    assets = [f"asset_{i}" for i in range(5)]
    
    # Create MultiIndex
    index = pd.MultiIndex.from_product([dates, assets], names=["date", "symbol"])
    
    # Create feature data
    data = pd.DataFrame(
        np.random.normal(0, 1, size=(50, 3)),
        index=index,
        columns=["feature_1", "feature_2", "feature_3"]
    )
    
    # Apply rank normalization
    normalized = rank_normalize(data)
    
    # Check that values are in [0, 1]
    assert (normalized >= 0).all().all()
    assert (normalized <= 1).all().all()
    
    # Check that each date has ranks that sum appropriately
    for date in dates:
        date_data = normalized.loc[date]
        for col in date_data.columns:
            # Check that ranks are between 0 and 1
            assert (date_data[col] >= 0).all()
            assert (date_data[col] <= 1).all()


def test_zscore_normalization():
    """Test z-score normalization."""
    # Create test data
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    assets = [f"asset_{i}" for i in range(5)]
    
    # Create MultiIndex
    index = pd.MultiIndex.from_product([dates, assets], names=["date", "symbol"])
    
    # Create feature data
    data = pd.DataFrame(
        np.random.normal(0, 1, size=(50, 3)),
        index=index,
        columns=["feature_1", "feature_2", "feature_3"]
    )
    
    # Apply z-score normalization
    normalized = zscore_normalize(data)
    
    # Check that mean is approximately 0 for each date
    for date in dates:
        date_data = normalized.loc[date]
        for col in date_data.columns:
            mean = date_data[col].mean()
            assert abs(mean) < 0.1  # Should be close to 0
    
    # Check that std is approximately 1 for each date
    for date in dates:
        date_data = normalized.loc[date]
        for col in date_data.columns:
            std = date_data[col].std()
            assert abs(std - 1) < 0.2  # Should be close to 1


def test_combined_normalization():
    """Test combined normalization."""
    # Create test data
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    assets = [f"asset_{i}" for i in range(5)]
    
    # Create MultiIndex
    index = pd.MultiIndex.from_product([dates, assets], names=["date", "symbol"])
    
    # Create feature data
    data = pd.DataFrame(
        np.random.normal(0, 1, size=(50, 3)),
        index=index,
        columns=["feature_1", "feature_2", "feature_3"]
    )
    
    # Create combined normalizer
    normalizer = CombinedNormalizer(
        include_raw=True,
        include_rank=True,
        include_zscore=True
    )
    
    # Apply normalization
    normalized = normalizer.normalize(data)
    
    # Check that we have 3x the number of columns (raw, rank, zscore)
    assert len(normalized.columns) == 9
    
    # Check column prefixes
    columns = list(normalized.columns)
    assert all(col.startswith(('raw_', 'rank_', 'zscore_')) for col in columns)


if __name__ == "__main__":
    test_rank_normalization()
    test_zscore_normalization()
    test_combined_normalization()
    print("All normalization tests passed!")
