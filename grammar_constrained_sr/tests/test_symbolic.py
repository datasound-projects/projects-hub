"""
Tests for symbolic regression module.
"""

import pytest
from src.symbolic.expressions import AlphaExpression, ExpressionAnalyzer


def test_expression_validation():
    """Test expression validation."""
    # Test valid expressions
    valid_expressions = [
        "x0 + x1",
        "x0 - x1",
        "x0 * x1",
        "x0 / x1",
        "abs(x0)",
        "neg(x0)",
        "abs(x0) + x1 * x2",
        "(x0 + x1) / (x2 - x3)"
    ]
    
    for expr in valid_expressions:
        alpha_expr = AlphaExpression(expression=expr, loss=0.1, complexity=3)
        assert alpha_expr.is_valid, f"Expression '{expr}' should be valid"
    
    # Test invalid expressions
    invalid_expressions = [
        "",  # Empty
        "x0 ** x1",  # Exponentiation not allowed
        "exp(x0)",  # Exp not allowed
        "log(x0)",  # Log not allowed
        "sin(x0)",  # Sin not allowed
        "x0 / 0",  # Division by zero
    ]
    
    for expr in invalid_expressions:
        alpha_expr = AlphaExpression(expression=expr, loss=0.1, complexity=3)
        assert not alpha_expr.is_valid, f"Expression '{expr}' should be invalid"


def test_sympy_conversion():
    """Test SymPy conversion."""
    # Test simple expression
    expr = AlphaExpression(
        expression="x0 + x1 * abs(x2)",
        loss=0.1,
        complexity=5,
        feature_names=["feature_0", "feature_1", "feature_2"]
    )
    
    # Should be valid
    assert expr.is_valid
    
    # Should have SymPy expression
    assert expr.sympy_expr is not None
    
    # Test that we can evaluate the SymPy expression
    if expr.sympy_expr:
        # Substitute values
        substituted = expr.sympy_expr.subs({
            'x_0': 1.0,
            'x_1': 2.0,
            'x_2': -3.0
        })
        
        # Evaluate
        result = float(substituted)
        expected = 1.0 + 2.0 * abs(-3.0)  # 1 + 6 = 7
        assert abs(result - expected) < 1e-10


def test_expression_analyzer():
    """Test expression analyzer."""
    analyzer = ExpressionAnalyzer()
    
    # Test analysis
    expr = AlphaExpression(
        expression="x0 + x1 * x2",
        loss=0.1,
        complexity=5,
        feature_names=["feature_0", "feature_1", "feature_2"]
    )
    
    analysis = analyzer.analyze_expression(expr)
    
    assert analysis['is_valid']
    assert analysis['complexity'] == 5
    assert len(analysis['used_features']) == 3


if __name__ == "__main__":
    test_expression_validation()
    test_sympy_conversion()
    test_expression_analyzer()
    print("All symbolic tests passed!")
