"""
Tests for code generation module.
"""

import pytest
from src.symbolic.expressions import AlphaExpression
from src.codegen.generator import generate_alpha_code


def test_code_generation():
    """Test code generation from alpha expressions."""
    # Create a simple alpha expression
    alpha_expr = AlphaExpression(
        expression="x0 + x1 * abs(x2)",
        loss=0.1,
        complexity=5,
        feature_names=["rolling_volatility_20", "rolling_mean_10", "hilbert_amplitude"]
    )
    
    # Generate Python code
    python_code = generate_alpha_code(alpha_expr, language="python")
    
    # Check that code contains the function definition
    assert "def alpha(" in python_code
    
    # Check that code contains the expression
    assert "x0 + x1 * abs(x2)" in python_code or "rolling_volatility_20" in python_code
    
    # Check that code contains all feature names
    for feature in alpha_expr.feature_names:
        assert feature in python_code
    
    # Generate NumPy code
    numpy_code = generate_alpha_code(alpha_expr, language="numpy")
    
    # Check that NumPy code has numpy import
    assert "import numpy as np" in numpy_code
    
    # Generate C code
    c_code = generate_alpha_code(alpha_expr, language="c")
    
    # Check that C code has function definition
    assert "double alpha(" in c_code
    
    # Check that C code has includes
    assert "#include" in c_code


def test_code_generation_with_complex_expression():
    """Test code generation with a more complex expression."""
    alpha_expr = AlphaExpression(
        expression="(x0 + x1) / (x2 - x3) * abs(x4)",
        loss=0.05,
        complexity=10,
        feature_names=["feature_0", "feature_1", "feature_2", "feature_3", "feature_4"]
    )
    
    # Generate code
    code = generate_alpha_code(alpha_expr, language="python")
    
    # Check that the expression is preserved
    assert "(" in code and ")" in code
    assert "/" in code
    assert "*" in code
    assert "abs" in code


if __name__ == "__main__":
    test_code_generation()
    test_code_generation_with_complex_expression()
    print("All code generation tests passed!")
