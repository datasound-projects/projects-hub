"""
Alpha expression handling and analysis.

This module provides classes and utilities for working with alpha expressions
discovered by symbolic regression.
"""

import re
import sympy as sp
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class AlphaExpression:
    """
    Represents an alpha expression discovered by symbolic regression.
    
    Contains the expression string, its loss, complexity, and other metadata.
    """
    
    # The expression as a string
    expression: str
    
    # Loss value from symbolic regression
    loss: float
    
    # Complexity of the expression (number of nodes)
    complexity: int
    
    # List of feature names used in the expression
    feature_names: List[str] = field(default_factory=list)
    
    # Rank of the expression (1 = best)
    rank: int = 1
    
    # SymPy expression (computed lazily)
    sympy_expr: Optional[sp.Expr] = None
    
    # Whether the expression is valid
    is_valid: bool = True
    
    # Validation message
    validation_message: str = ""
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize the expression and validate it."""
        # Validate the expression
        self.is_valid, self.validation_message = self.validate()
        
        # Parse to SymPy if valid
        if self.is_valid:
            try:
                self.sympy_expr = self.to_sympy()
            except Exception as e:
                logger.warning(f"Failed to parse expression to SymPy: {str(e)}")
                self.is_valid = False
                self.validation_message = f"SymPy parsing failed: {str(e)}"
    
    def validate(self) -> Tuple[bool, str]:
        """
        Validate the expression.
        
        Returns:
            Tuple of (is_valid, message)
        """
        # Check for empty expression
        if not self.expression or not self.expression.strip():
            return False, "Empty expression"
        
        # Check for division by zero
        if "/ 0" in self.expression or "/0" in self.expression:
            return False, "Division by zero in expression"
        
        # Check for invalid characters
        if not re.match(r'^[\[\]\d\s+\-*/().,absneg]+$', self.expression):
            return False, f"Invalid characters in expression: {self.expression}"
        
        # Check for balanced parentheses
        if self.expression.count('(') != self.expression.count(')'):
            return False, "Unbalanced parentheses"
        
        # Check for allowed operators only
        allowed_operators = ['+', '-', '*', '/', 'abs', 'neg']
        for op in allowed_operators:
            if op in self.expression:
                pass  # Operator is allowed
        
        # Check for disallowed operators
        disallowed = ['**', '^', 'exp', 'log', 'sin', 'cos', 'tan', 'sqrt']
        for op in disallowed:
            if op in self.expression:
                return False, f"Disallowed operator '{op}' in expression"
        
        return True, "Valid"
    
    def to_sympy(self) -> sp.Expr:
        """
        Convert the expression to a SymPy expression.
        
        Returns:
            SymPy expression
        """
        # Replace feature names with symbols
        expr_str = self.expression
        
        # Create symbol mapping
        symbols = {}
        for i, name in enumerate(self.feature_names):
            # Replace feature names with symbols
            symbol_name = f"x_{i}"
            symbols[symbol_name] = sp.Symbol(symbol_name)
            expr_str = expr_str.replace(name, symbol_name)
        
        # Replace operators
        expr_str = expr_str.replace('abs', 'Abs')
        expr_str = expr_str.replace('neg', '-')
        
        # Parse the expression
        try:
            sympy_expr = sp.sympify(expr_str)
            return sympy_expr
        except Exception as e:
            logger.error(f"Failed to parse '{expr_str}' to SymPy: {str(e)}")
            raise
    
    def to_lambda(self) -> callable:
        """
        Convert the expression to a lambda function.
        
        Returns:
            Lambda function that takes feature values and returns alpha
        """
        if not self.sympy_expr:
            raise ValueError("SymPy expression not available")
        
        # Get the symbols from the expression
        symbols = self.sympy_expr.free_symbols
        
        # Create a lambda function
        def alpha_function(*args, **kwargs):
            # Create symbol mapping
            symbol_map = {}
            for i, symbol in enumerate(symbols):
                symbol_map[symbol] = args[i] if i < len(args) else kwargs.get(str(symbol), 0)
            
            # Evaluate the expression
            return float(self.sympy_expr.subs(symbol_map))
        
        return alpha_function
    
    def to_code(self, 
                language: str = 'python',
                function_name: str = 'alpha') -> str:
        """
        Generate code for the expression.
        
        Args:
            language: Target language ('python', 'numpy', 'c')
            function_name: Name of the function
            
        Returns:
            Code string
        """
        if not self.sympy_expr:
            raise ValueError("SymPy expression not available")
        
        if language == 'python':
            return self._to_python_code(function_name)
        elif language == 'numpy':
            return self._to_numpy_code(function_name)
        elif language == 'c':
            return self._to_c_code(function_name)
        else:
            raise ValueError(f"Unsupported language: {language}")
    
    def _to_python_code(self, function_name: str) -> str:
        """Generate Python code."""
        # Get the expression string
        expr_str = self.expression
        
        # Create parameter list
        params = ", ".join(self.feature_names)
        
        # Create function
        code = f"def {function_name}({params}):\n"
        code += f"    return {expr_str}\n"
        
        return code
    
    def _to_numpy_code(self, function_name: str) -> str:
        """Generate NumPy-optimized code."""
        # Get the expression string
        expr_str = self.expression
        
        # Create parameter list
        params = ", ".join(self.feature_names)
        
        # Create function with NumPy operations
        code = f"import numpy as np\n\n"
        code += f"def {function_name}({params}):\n"
        code += f"    return {expr_str}\n"
        
        return code
    
    def _to_c_code(self, function_name: str) -> str:
        """Generate C code using SymPy."""
        if not self.sympy_expr:
            raise ValueError("SymPy expression not available")
        
        # Generate C code using SymPy
        c_code = sp.ccode(self.sympy_expr)
        
        # Create function wrapper
        params = ", ".join([f"double {name}" for name in self.feature_names])
        
        code = f"double {function_name}({params}) {{\n"
        code += f"    return {c_code};\n"
        code += f"}}\n"
        
        return code
    
    def get_used_features(self) -> List[str]:
        """
        Get the list of features used in the expression.
        
        Returns:
            List of feature names
        """
        used_features = []
        
        for name in self.feature_names:
            if name in self.expression:
                used_features.append(name)
        
        return used_features
    
    def complexity_breakdown(self) -> Dict[str, int]:
        """
        Get a breakdown of the expression complexity.
        
        Returns:
            Dictionary with complexity metrics
        """
        if not self.sympy_expr:
            return {'total': self.complexity}
        
        # Count operators
        operators = {
            '+': 0,
            '-': 0,
            '*': 0,
            '/': 0,
            'abs': 0,
            'neg': 0
        }
        
        # Traverse the expression tree
        self._count_operators(self.sympy_expr, operators)
        
        # Count symbols (features)
        n_features = len(self.sympy_expr.free_symbols)
        
        return {
            'total': self.complexity,
            'operators': operators,
            'n_features': n_features,
            'n_constants': 0  # We don't have constants in our grammar
        }
    
    def _count_operators(self, expr: sp.Expr, operators: Dict[str, int]):
        """Recursively count operators in an expression."""
        if expr.is_Add:
            operators['+'] += 1
            for arg in expr.args:
                self._count_operators(arg, operators)
        elif expr.is_Mul:
            operators['*'] += 1
            for arg in expr.args:
                self._count_operators(arg, operators)
        elif expr.is_Pow:
            # We don't have exponentiation in our grammar
            pass
        elif expr.is_Abs:
            operators['abs'] += 1
            self._count_operators(expr.args[0], operators)
        elif expr.is_Negative:
            operators['neg'] += 1
            self._count_operators(-expr, operators)
        elif expr.is_Div:
            operators['/'] += 1
            self._count_operators(expr.args[0], operators)
            self._count_operators(expr.args[1], operators)
        elif expr.is_Sub:
            operators['-'] += 1
            self._count_operators(expr.args[0], operators)
            self._count_operators(expr.args[1], operators)


class ExpressionAnalyzer:
    """
    Analyzer for alpha expressions.
    
    Provides utilities for analyzing, simplifying, and validating expressions.
    """
    
    def __init__(self):
        """Initialize the analyzer."""
        pass
    
    def analyze_expression(self, 
                         expression: Union[str, AlphaExpression]) -> Dict[str, Any]:
        """
        Analyze an expression.
        
        Args:
            expression: Expression string or AlphaExpression
            
        Returns:
            Dictionary with analysis results
        """
        if isinstance(expression, str):
            alpha_expr = AlphaExpression(expression, 0, 0)
        else:
            alpha_expr = expression
        
        analysis = {
            'expression': alpha_expr.expression,
            'is_valid': alpha_expr.is_valid,
            'validation_message': alpha_expr.validation_message,
            'complexity': alpha_expr.complexity,
            'used_features': alpha_expr.get_used_features(),
            'complexity_breakdown': alpha_expr.complexity_breakdown()
        }
        
        return analysis
    
    def simplify_expression(self, 
                           expression: Union[str, AlphaExpression]) -> AlphaExpression:
        """
        Simplify an expression.
        
        Args:
            expression: Expression string or AlphaExpression
            
        Returns:
            Simplified AlphaExpression
        """
        if isinstance(expression, str):
            alpha_expr = AlphaExpression(expression, 0, 0)
        else:
            alpha_expr = expression
        
        if not alpha_expr.sympy_expr:
            return alpha_expr
        
        # Simplify using SymPy
        simplified_expr = sp.simplify(alpha_expr.sympy_expr)
        
        # Convert back to string
        simplified_str = str(simplified_expr)
        
        # Create new AlphaExpression with simplified string
        simplified_alpha = AlphaExpression(
            expression=simplified_str,
            loss=alpha_expr.loss,
            complexity=len(simplified_str),
            feature_names=alpha_expr.feature_names,
            rank=alpha_expr.rank
        )
        
        return simplified_alpha
    
    def validate_expression(self, 
                          expression: Union[str, AlphaExpression]) -> Tuple[bool, str]:
        """
        Validate an expression.
        
        Args:
            expression: Expression string or AlphaExpression
            
        Returns:
            Tuple of (is_valid, message)
        """
        if isinstance(expression, str):
            alpha_expr = AlphaExpression(expression, 0, 0)
        else:
            alpha_expr = expression
        
        return alpha_expr.is_valid, alpha_expr.validation_message
    
    def compare_expressions(self, 
                          expr1: Union[str, AlphaExpression],
                          expr2: Union[str, AlphaExpression]) -> Dict[str, Any]:
        """
        Compare two expressions.
        
        Args:
            expr1: First expression
            expr2: Second expression
            
        Returns:
            Dictionary with comparison results
        """
        alpha1 = expr1 if isinstance(expr1, AlphaExpression) else AlphaExpression(expr1, 0, 0)
        alpha2 = expr2 if isinstance(expr2, AlphaExpression) else AlphaExpression(expr2, 0, 0)
        
        comparison = {
            'expr1': alpha1.expression,
            'expr2': alpha2.expression,
            'same_expression': alpha1.expression == alpha2.expression,
            'same_features': set(alpha1.get_used_features()) == set(alpha2.get_used_features()),
            'complexity_diff': alpha1.complexity - alpha2.complexity,
            'loss_diff': alpha1.loss - alpha2.loss
        }
        
        return comparison


def simplify_expression(expression: Union[str, AlphaExpression]) -> AlphaExpression:
    """
    Convenience function to simplify an expression.
    
    Args:
        expression: Expression string or AlphaExpression
        
    Returns:
        Simplified AlphaExpression
    """
    analyzer = ExpressionAnalyzer()
    return analyzer.simplify_expression(expression)


def validate_expression(expression: Union[str, AlphaExpression]) -> Tuple[bool, str]:
    """
    Convenience function to validate an expression.
    
    Args:
        expression: Expression string or AlphaExpression
        
    Returns:
        Tuple of (is_valid, message)
    """
    analyzer = ExpressionAnalyzer()
    return analyzer.validate_expression(expression)
