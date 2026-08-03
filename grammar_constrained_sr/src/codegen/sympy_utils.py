"""
SymPy utilities for code generation.

This module provides utilities for converting alpha expressions to SymPy
and then to various programming languages.
"""

import sympy as sp
import re
from typing import Dict, List, Optional, Union, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class SymPyUtils:
    """
    Utilities for working with SymPy expressions.
    """
    
    @staticmethod
    def expression_to_sympy(expression: str, 
                           feature_names: List[str]) -> sp.Expr:
        """
        Convert an expression string to a SymPy expression.
        
        Args:
            expression: Expression string
            feature_names: List of feature names used in the expression
            
        Returns:
            SymPy expression
        """
        # Replace feature names with symbols
        expr_str = expression
        
        # Create symbol mapping
        symbols = {}
        for i, name in enumerate(feature_names):
            # Clean the feature name for use as a symbol
            clean_name = SymPyUtils._clean_symbol_name(name)
            symbol = sp.Symbol(clean_name)
            symbols[clean_name] = symbol
            
            # Replace feature name in expression
            expr_str = expr_str.replace(name, clean_name)
        
        # Replace operators
        expr_str = expr_str.replace('abs', 'Abs')
        expr_str = expr_str.replace('neg', '-')
        
        # Parse the expression
        try:
            sympy_expr = sp.sympify(expr_str)
            return sympy_expr
        except Exception as e:
            logger.error(f"Failed to parse '{expr_str}' to SymPy: {str(e)}")
            raise ValueError(f"Invalid expression: {expression}")
    
    @staticmethod
    def _clean_symbol_name(name: str) -> str:
        """Clean a feature name for use as a SymPy symbol."""
        # Replace invalid characters
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        
        # Ensure it doesn't start with a number
        if clean_name[0].isdigit():
            clean_name = f"_{clean_name}"
        
        # Remove consecutive underscores
        clean_name = re.sub(r'_+', '_', clean_name)
        
        # Remove leading/trailing underscores
        clean_name = clean_name.strip('_')
        
        return clean_name
    
    @staticmethod
    def sympy_to_python(sympy_expr: sp.Expr,
                       function_name: str = "alpha",
                       feature_names: Optional[List[str]] = None) -> str:
        """
        Convert a SymPy expression to Python code.
        
        Args:
            sympy_expr: SymPy expression
            function_name: Name of the generated function
            feature_names: List of feature names (for parameter list)
            
        Returns:
            Python code string
        """
        # Get the expression string
        expr_str = str(sympy_expr)
        
        # Replace SymPy functions with Python equivalents
        expr_str = expr_str.replace('Abs', 'abs')
        
        # Create parameter list
        if feature_names:
            params = ", ".join(feature_names)
        else:
            # Extract symbols from expression
            symbols = sympy_expr.free_symbols
            params = ", ".join([str(s) for s in symbols])
        
        # Create function
        code = f"def {function_name}({params}):\n"
        code += f"    return {expr_str}\n"
        
        return code
    
    @staticmethod
    def sympy_to_numpy(sympy_expr: sp.Expr,
                      function_name: str = "alpha",
                      feature_names: Optional[List[str]] = None) -> str:
        """
        Convert a SymPy expression to NumPy-optimized code.
        
        Args:
            sympy_expr: SymPy expression
            function_name: Name of the generated function
            feature_names: List of feature names
            
        Returns:
            NumPy code string
        """
        # Get the expression string
        expr_str = str(sympy_expr)
        
        # Replace SymPy functions with NumPy equivalents
        expr_str = expr_str.replace('Abs', 'np.abs')
        
        # Create parameter list
        if feature_names:
            params = ", ".join(feature_names)
        else:
            symbols = sympy_expr.free_symbols
            params = ", ".join([str(s) for s in symbols])
        
        # Create function with NumPy import
        code = "import numpy as np\n\n"
        code += f"def {function_name}({params}):\n"
        code += f"    return {expr_str}\n"
        
        return code
    
    @staticmethod
    def sympy_to_c(sympy_expr: sp.Expr,
                   function_name: str = "alpha",
                   feature_names: Optional[List[str]] = None) -> str:
        """
        Convert a SymPy expression to C code.
        
        Args:
            sympy_expr: SymPy expression
            function_name: Name of the generated function
            feature_names: List of feature names
            
        Returns:
            C code string
        """
        # Use SymPy's ccode function
        c_expr = sp.ccode(sympy_expr)
        
        # Create parameter list
        if feature_names:
            params = ", ".join([f"double {name}" for name in feature_names])
        else:
            symbols = sympy_expr.free_symbols
            params = ", ".join([f"double {s}" for s in symbols])
        
        # Create function
        code = f"double {function_name}({params}) {{\n"
        code += f"    return {c_expr};\n"
        code += f"}}\n"
        
        return code
    
    @staticmethod
    def sympy_to_lambda(sympy_expr: sp.Expr,
                       feature_names: List[str]) -> callable:
        """
        Convert a SymPy expression to a lambda function.
        
        Args:
            sympy_expr: SymPy expression
            feature_names: List of feature names
            
        Returns:
            Lambda function
        """
        # Get the expression string
        expr_str = str(sympy_expr)
        
        # Replace SymPy functions
        expr_str = expr_str.replace('Abs', 'abs')
        
        # Create lambda function
        params = ", ".join(feature_names)
        
        try:
            lambda_func = eval(f"lambda {params}: {expr_str}")
            return lambda_func
        except Exception as e:
            logger.error(f"Failed to create lambda function: {str(e)}")
            raise
    
    @staticmethod
    def simplify_expression(sympy_expr: sp.Expr) -> sp.Expr:
        """
        Simplify a SymPy expression.
        
        Args:
            sympy_expr: SymPy expression
            
        Returns:
            Simplified SymPy expression
        """
        return sp.simplify(sympy_expr)
    
    @staticmethod
    def get_expression_info(sympy_expr: sp.Expr) -> Dict[str, Any]:
        """
        Get information about a SymPy expression.
        
        Args:
            sympy_expr: SymPy expression
            
        Returns:
            Dictionary with expression information
        """
        info = {
            'expression': str(sympy_expr),
            'free_symbols': [str(s) for s in sympy_expr.free_symbols],
            'is_polynomial': sympy_expr.is_polynomial(),
            'is_rational': sympy_expr.is_rational_function(),
            'degree_list': sympy_expr.as_poly().degree_list() if sympy_expr.is_polynomial() else None
        }
        
        return info


def expression_to_sympy(expression: str, 
                        feature_names: List[str]) -> sp.Expr:
    """
    Convenience function to convert expression to SymPy.
    
    Args:
        expression: Expression string
        feature_names: List of feature names
        
    Returns:
        SymPy expression
    """
    return SymPyUtils.expression_to_sympy(expression, feature_names)


def sympy_to_python(sympy_expr: sp.Expr,
                   function_name: str = "alpha",
                   feature_names: Optional[List[str]] = None) -> str:
    """
    Convenience function to convert SymPy to Python.
    
    Args:
        sympy_expr: SymPy expression
        function_name: Function name
        feature_names: Feature names
        
    Returns:
        Python code string
    """
    return SymPyUtils.sympy_to_python(sympy_expr, function_name, feature_names)


def sympy_to_numpy(sympy_expr: sp.Expr,
                   function_name: str = "alpha",
                   feature_names: Optional[List[str]] = None) -> str:
    """
    Convenience function to convert SymPy to NumPy code.
    
    Args:
        sympy_expr: SymPy expression
        function_name: Function name
        feature_names: Feature names
        
    Returns:
        NumPy code string
    """
    return SymPyUtils.sympy_to_numpy(sympy_expr, function_name, feature_names)


def sympy_to_c(sympy_expr: sp.Expr,
               function_name: str = "alpha",
               feature_names: Optional[List[str]] = None) -> str:
    """
    Convenience function to convert SymPy to C code.
    
    Args:
        sympy_expr: SymPy expression
        function_name: Function name
        feature_names: Feature names
        
    Returns:
        C code string
    """
    return SymPyUtils.sympy_to_c(sympy_expr, function_name, feature_names)
