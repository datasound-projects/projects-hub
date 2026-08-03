"""
Tests for feature primitives.
"""

import pandas as pd
import numpy as np
import pytest
from src.features.volatility import RollingVolatility, RollingKurtosis
from src.features.autocorrelation import RollingAutocorrelation


def test_volatility_primitives():
    """Test volatility primitives."""
    # Create test data
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    returns = pd.DataFrame(
        np.random.normal(0, 0.02, size=(100, 10)),
        index=dates,
        columns=[f"asset_{i}" for i in range(10)]
    )
    
    # Test RollingVolatility
    vol_calc = RollingVolatility()
    volatility = vol_calc.compute(returns, window=20)
    
    # Check output shape
    assert volatility.shape == returns.shape
    
    # Check that volatility is positive
    assert (volatility >= 0).all().all()
    
    # Check that first 19 rows are NaN (window=20)
    assert volatility.iloc[:19].isna().all().all()
    
    # Test RollingKurtosis
    kurtosis_calc = RollingKurtosis()
    kurtosis = kurtosis_calc.compute(returns, window=20)
    
    # Check output shape
    assert kurtosis.shape == returns.shape
    
    # Check that first 19 rows are NaN
    assert kurtosis.iloc[:19].isna().all().all()


def test_autocorrelation_primitives():
    """Test autocorrelation primitives."""
    # Create test data with some autocorrelation
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    
    # Create data with positive autocorrelation
    data = pd.DataFrame(index=dates)
    for i in range(5):
        # AR(1) process with positive autocorrelation
        x = np.random.normal(0, 1, 100)
        for t in range(1, 100):
            x[t] = 0.7 * x[t-1] + np.random.normal(0, 0.5)
        data[f"asset_{i}"] = x
    
    # Test RollingAutocorrelation
    ac_calc = RollingAutocorrelation(lag=1)
    autocorr = ac_calc.compute(data, window=20)
    
    # Check output shape
    assert autocorr.shape == data.shape
    
    # Check that first 19 rows are NaN
    assert autocorr.iloc[:19].isna().all().all()
    
    # Check that autocorrelation values are in [-1, 1]
    valid_ac = autocorr.dropna()
    assert (valid_ac >= -1).all().all()
    assert (valid_ac <= 1).all().all()


if __name__ == "__main__":
    test_volatility_primitives()
    test_autocorrelation_primitives()
    print("All feature tests passed!")
