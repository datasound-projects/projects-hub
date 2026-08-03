"""
Code generator for alpha expressions and trading strategies.

This module provides functionality to generate deployable code from
discovered alpha expressions.
"""

import sympy as sp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
import logging

from .sympy_utils import SymPyUtils, expression_to_sympy
from ..symbolic.expressions import AlphaExpression

logger = logging.getLogger(__name__)


@dataclass
class CodeGenerationConfig:
    """Configuration for code generation."""
    # Target language
    language: str = "python"
    
    # Function name
    function_name: str = "alpha"
    
    # Whether to include type hints
    include_type_hints: bool = True
    
    # Whether to include docstring
    include_docstring: bool = True
    
    # Whether to include error handling
    include_error_handling: bool = True
    
    # Whether to optimize for performance
    optimize: bool = True


class CodeGenerator:
    """
    Generator for creating deployable code from alpha expressions.
    
    Supports multiple target languages and can generate:
    - Alpha function code
    - Portfolio construction code
    - Complete backtest code
    """
    
    def __init__(self, 
                 config: Optional[CodeGenerationConfig] = None):
        """
        Initialize the CodeGenerator.
        
        Args:
            config: Code generation configuration
        """
        self.config = config or CodeGenerationConfig()
    
    def generate_alpha_code(self, 
                           expression: Union[str, AlphaExpression],
                           feature_names: Optional[List[str]] = None) -> str:
        """
        Generate code for an alpha expression.
        
        Args:
            expression: Expression string or AlphaExpression
            feature_names: List of feature names (if not in AlphaExpression)
            
        Returns:
            Generated code string
        """
        if isinstance(expression, AlphaExpression):
            expr_str = expression.expression
            if feature_names is None:
                feature_names = expression.feature_names
        else:
            expr_str = expression
            if feature_names is None:
                raise ValueError("feature_names must be provided for string expressions")
        
        # Convert to SymPy
        sympy_expr = expression_to_sympy(expr_str, feature_names)
        
        # Generate code based on target language
        if self.config.language == "python":
            code = self._generate_python_alpha_code(sympy_expr, feature_names)
        elif self.config.language == "numpy":
            code = self._generate_numpy_alpha_code(sympy_expr, feature_names)
        elif self.config.language == "c":
            code = self._generate_c_alpha_code(sympy_expr, feature_names)
        else:
            raise ValueError(f"Unsupported language: {self.config.language}")
        
        return code
    
    def _generate_python_alpha_code(self, 
                                   sympy_expr: sp.Expr,
                                   feature_names: List[str]) -> str:
        """Generate Python code for alpha function."""
        # Generate basic function
        code = SymPyUtils.sympy_to_python(
            sympy_expr,
            self.config.function_name,
            feature_names
        )
        
        # Add type hints if requested
        if self.config.include_type_hints:
            params = ", ".join([f"{name}: float" for name in feature_names])
            return_type = " -> float"
            
            # Replace the parameter line
            code = code.replace(
                f"def {self.config.function_name}({', '.join(feature_names)}):",
                f"def {self.config.function_name}({params}){return_type}:"
            )
        
        # Add docstring if requested
        if self.config.include_docstring:
            docstring = f'"""\n    Compute alpha value from features.\n    \n    Args:\n'
            for name in feature_names:
                docstring += f'        {name}: Feature value\n'
            docstring += f'    \n    Returns:\n        Alpha value\n    """\n'
            
            # Insert docstring after function definition
            lines = code.split('\n')
            lines.insert(1, docstring)
            code = '\n'.join(lines)
        
        # Add error handling if requested
        if self.config.include_error_handling:
            # Wrap the return statement in try-except
            lines = code.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('return '):
                    indent = len(line) - len(line.lstrip())
                    lines[i] = ' ' * indent + 'try:\n' + ' ' * (indent + 4) + line.strip()
                    lines.insert(i + 1, ' ' * indent + 'except Exception as e:\n' + 
                                ' ' * (indent + 4) + 'print(f"Error in alpha calculation: {{str(e)}}")\n' + 
                                ' ' * (indent + 4) + 'return 0.0')
                    break
            code = '\n'.join(lines)
        
        return code
    
    def _generate_numpy_alpha_code(self, 
                                  sympy_expr: sp.Expr,
                                  feature_names: List[str]) -> str:
        """Generate NumPy-optimized code for alpha function."""
        code = SymPyUtils.sympy_to_numpy(
            sympy_expr,
            self.config.function_name,
            feature_names
        )
        
        # Add type hints
        if self.config.include_type_hints:
            params = ", ".join([f"{name}: np.ndarray" for name in feature_names])
            return_type = " -> np.ndarray"
            
            code = code.replace(
                f"def {self.config.function_name}({', '.join(feature_names)}):",
                f"def {self.config.function_name}({params}){return_type}:"
            )
        
        # Add docstring
        if self.config.include_docstring:
            docstring = f'"""\n    Compute alpha values from feature arrays.\n    \n    Args:\n'
            for name in feature_names:
                docstring += f'        {name}: Array of feature values\n'
            docstring += f'    \n    Returns:\n        Array of alpha values\n    """\n'
            
            lines = code.split('\n')
            lines.insert(2, docstring)  # After the import
            code = '\n'.join(lines)
        
        return code
    
    def _generate_c_alpha_code(self, 
                               sympy_expr: sp.Expr,
                               feature_names: List[str]) -> str:
        """Generate C code for alpha function."""
        code = SymPyUtils.sympy_to_c(
            sympy_expr,
            self.config.function_name,
            feature_names
        )
        
        # Add header guard and includes
        header_guard = self.config.function_name.upper() + "_H"
        
        full_code = f"#ifndef {header_guard}\n"
        full_code += f"#define {header_guard}\n\n"
        full_code += "#include <math.h>\n\n"
        full_code += code
        full_code += f"\n#endif  // {header_guard}\n"
        
        return full_code
    
    def generate_portfolio_code(self, 
                              alpha_expression: Union[str, AlphaExpression],
                              feature_names: Optional[List[str]] = None,
                              n_quintiles: int = 5) -> str:
        """
        Generate code for constructing a portfolio from an alpha.
        
        Args:
            alpha_expression: Alpha expression
            feature_names: List of feature names
            n_quintiles: Number of quintiles for long-short
            
        Returns:
            Generated code string
        """
        # Generate alpha function
        alpha_code = self.generate_alpha_code(alpha_expression, feature_names)
        
        # Generate portfolio construction code
        if self.config.language == "python":
            portfolio_code = self._generate_python_portfolio_code(
                alpha_expression, feature_names, n_quintiles
            )
        elif self.config.language == "numpy":
            portfolio_code = self._generate_numpy_portfolio_code(
                alpha_expression, feature_names, n_quintiles
            )
        else:
            portfolio_code = ""
        
        return f"{alpha_code}\n\n{portfolio_code}"
    
    def _generate_python_portfolio_code(self, 
                                         alpha_expression: Union[str, AlphaExpression],
                                         feature_names: Optional[List[str]],
                                         n_quintiles: int) -> str:
        """Generate Python code for portfolio construction."""
        if isinstance(alpha_expression, AlphaExpression):
            func_name = self.config.function_name
        else:
            func_name = self.config.function_name
        
        code = f"def construct_portfolio(features: pd.DataFrame) -> pd.Series:\n"
        code += f'    """\n'
        code += f'    Construct a long-short portfolio from alpha signals.\n\n'
        code += f'    Args:\n'
        code += f'        features: DataFrame with feature values\n\n'
        code += f'    Returns:\n'
        code += f'        Portfolio weights\n'
        code += f'    """\n'
        code += f'    # Compute alpha values\n'
        code += f'    alpha_values = features.apply(lambda row: {func_name}(' + 
                ', '.join([f"row['{name}']" for name in feature_names]) + '), axis=1)\n'
        code += f'\n'
        code += f'    # Rank assets by alpha\n'
        code += f'    ranked = alpha_values.rank(method="average")\n'
        code += f'\n'
        code += f'    # Create long-short portfolio\n'
        code += f'    n = len(ranked)\n'
        code += f'    n_long = n // {n_quintiles}\n'
        code += f'    n_short = n // {n_quintiles}\n'
        code += f'\n'
        code += f'    weights = pd.Series(0.0, index=ranked.index)\n'
        code += f'    weights[ranked >= n - n_long] = 1.0 / n_long  # Long top quintile\n'
        code += f'    weights[ranked <= n_short] = -1.0 / n_short  # Short bottom quintile\n'
        code += f'\n'
        code += f'    return weights\n'
        
        return code
    
    def _generate_numpy_portfolio_code(self, 
                                        alpha_expression: Union[str, AlphaExpression],
                                        feature_names: Optional[List[str]],
                                        n_quintiles: int) -> str:
        """Generate NumPy code for portfolio construction."""
        if isinstance(alpha_expression, AlphaExpression):
            func_name = self.config.function_name
        else:
            func_name = self.config.function_name
        
        code = f"def construct_portfolio(features: np.ndarray) -> np.ndarray:\n"
        code += f'    """\n'
        code += f'    Construct a long-short portfolio from alpha signals.\n\n'
        code += f'    Args:\n'
        code += f'        features: Array of feature values (n_assets x n_features)\n\n'
        code += f'    Returns:\n'
        code += f'        Portfolio weights (n_assets,)\n'
        code += f'    """\n'
        code += f'    # Compute alpha values\n'
        code += f'    alpha_values = np.array([{func_name}(' + 
                ', '.join([f"row[{i}]" for i in range(len(feature_names))]) + 
                ') for row in features])\n'
        code += f'\n'
        code += f'    # Rank assets by alpha\n'
        code += f'    ranked = np.argsort(np.argsort(alpha_values))\n'
        code += f'\n'
        code += f'    # Create long-short portfolio\n'
        code += f'    n = len(ranked)\n'
        code += f'    n_long = n // {n_quintiles}\n'
        code += f'    n_short = n // {n_quintiles}\n'
        code += f'\n'
        code += f'    weights = np.zeros(n)\n'
        code += f'    weights[ranked >= n - n_long] = 1.0 / n_long  # Long top quintile\n'
        code += f'    weights[ranked <= n_short] = -1.0 / n_short  # Short bottom quintile\n'
        code += f'\n'
        code += f'    return weights\n'
        
        return code
    
    def generate_backtest_code(self, 
                            alpha_expression: Union[str, AlphaExpression],
                            feature_names: Optional[List[str]] = None,
                            n_quintiles: int = 5) -> str:
        """
        Generate complete backtest code.
        
        Args:
            alpha_expression: Alpha expression
            feature_names: List of feature names
            n_quintiles: Number of quintiles
            
        Returns:
            Complete backtest code
        """
        # Generate alpha and portfolio code
        portfolio_code = self.generate_portfolio_code(
            alpha_expression, feature_names, n_quintiles
        )
        
        # Add backtest function
        if self.config.language == "python":
            backtest_code = self._generate_python_backtest_code(
                alpha_expression, feature_names, n_quintiles
            )
        else:
            backtest_code = ""
        
        # Combine all code
        full_code = f"""
# Alpha Backtest Code
# Generated by Grammar-Constrained Symbolic Regression for Alpha Discovery

import pandas as pd
import numpy as np

{portfolio_code}

{backtest_code}
"""
        
        return full_code
    
    def _generate_python_backtest_code(self, 
                                       alpha_expression: Union[str, AlphaExpression],
                                       feature_names: Optional[List[str]],
                                       n_quintiles: int) -> str:
        """Generate Python backtest code."""
        code = f"""
def run_backtest(features: pd.DataFrame, 
                forward_returns: pd.Series,
                start_date: Optional[str] = None,
                end_date: Optional[str] = None) -> Dict[str, Any]:
    \"\"\"
    Run a backtest of the alpha strategy.
    
    Args:
        features: DataFrame with feature values
        forward_returns: Series with forward returns
        start_date: Optional start date
        end_date: Optional end date
        
    Returns:
        Dictionary with backtest results
    \"\"\"
    # Filter by date range
    if start_date:
        features = features.loc[start_date:]
        forward_returns = forward_returns.loc[start_date:]
    if end_date:
        features = features.loc[:end_date]
        forward_returns = forward_returns.loc[:end_date]
    
    # Align indices
    common_index = features.index.intersection(forward_returns.index)
    features = features.loc[common_index]
    forward_returns = forward_returns.loc[common_index]
    
    # Compute portfolio weights for each date
    portfolio_weights = []
    portfolio_returns = []
    
    dates = features.index.unique()
    for date in dates:
        date_features = features.loc[date]
        date_returns = forward_returns.loc[date]
        
        weights = construct_portfolio(date_features)
        portfolio_weights.append(weights)
        
        # Compute portfolio return
        portfolio_return = (weights * date_returns).sum()
        portfolio_returns.append(portfolio_return)
    
    # Compute performance metrics
    portfolio_returns = pd.Series(portfolio_returns, index=dates)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    
    # Annualized return
    total_return = cumulative_returns.iloc[-1] - 1
    n_periods = len(portfolio_returns)
    annualized_return = (1 + total_return) ** (252 / n_periods) - 1
    
    # Annualized volatility
    annualized_vol = portfolio_returns.std() * np.sqrt(252)
    
    # Sharpe ratio
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
    
    # Max drawdown
    running_max = cumulative_returns.cummax()
    drawdowns = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdowns.min()
    
    return {
        'portfolio_returns': portfolio_returns,
        'cumulative_returns': cumulative_returns,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_vol,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'portfolio_weights': pd.DataFrame(portfolio_weights, index=dates)
    }
"""
        
        return code


def generate_alpha_code(expression: Union[str, AlphaExpression],
                       feature_names: Optional[List[str]] = None,
                       language: str = "python") -> str:
    """
    Convenience function to generate alpha code.
    
    Args:
        expression: Expression string or AlphaExpression
        feature_names: List of feature names
        language: Target language
        
    Returns:
        Generated code string
    """
    config = CodeGenerationConfig(language=language)
    generator = CodeGenerator(config)
    return generator.generate_alpha_code(expression, feature_names)


def generate_portfolio_code(expression: Union[str, AlphaExpression],
                           feature_names: Optional[List[str]] = None,
                           n_quintiles: int = 5,
                           language: str = "python") -> str:
    """
    Convenience function to generate portfolio code.
    
    Args:
        expression: Expression string or AlphaExpression
        feature_names: List of feature names
        n_quintiles: Number of quintiles
        language: Target language
        
    Returns:
        Generated code string
    """
    config = CodeGenerationConfig(language=language)
    generator = CodeGenerator(config)
    return generator.generate_portfolio_code(expression, feature_names, n_quintiles)


def generate_backtest_code(expression: Union[str, AlphaExpression],
                          feature_names: Optional[List[str]] = None,
                          n_quintiles: int = 5,
                          language: str = "python") -> str:
    """
    Convenience function to generate backtest code.
    
    Args:
        expression: Expression string or AlphaExpression
        feature_names: List of feature names
        n_quintiles: Number of quintiles
        language: Target language
        
    Returns:
        Generated code string
    """
    config = CodeGenerationConfig(language=language)
    generator = CodeGenerator(config)
    return generator.generate_backtest_code(expression, feature_names, n_quintiles)
