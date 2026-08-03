"""
Symbolic regression for alpha discovery using PySR.

This module implements the symbolic regression stage of the pipeline,
using PySR with grammar constraints to discover alpha expressions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
import logging

try:
    import pysr
    PYSRAVAILABLE = True
except ImportError:
    PYSRAVAILABLE = False
    logging.warning("PySR not available. Symbolic regression will not work.")

from .pysr_config import PySRConfig, get_default_config, config_to_dict
from .expressions import AlphaExpression, ExpressionAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class SymbolicRegressionResult:
    """Result of symbolic regression."""
    expressions: List[AlphaExpression]
    best_expression: Optional[AlphaExpression]
    pareto_front: Optional[pd.DataFrame] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    config: Optional[PySRConfig] = None


class SymbolicRegressor:
    """
    Main class for symbolic regression using PySR.
    
    Enforces grammar constraints and manages the search for alpha expressions.
    """
    
    def __init__(self, 
                 config: Optional[PySRConfig] = None):
        """
        Initialize the SymbolicRegressor.
        
        Args:
            config: PySR configuration
        """
        if not PYSRAVAILABLE:
            raise ImportError("PySR is required for symbolic regression. Install with: pip install pysr")
        
        self.config = config or get_default_config()
        self.analyzer = ExpressionAnalyzer()
        
    def fit(self, 
            X: pd.DataFrame,
            y: pd.Series,
            feature_names: Optional[List[str]] = None) -> SymbolicRegressionResult:
        """
        Fit symbolic regression model to data.
        
        Args:
            X: Feature matrix (n_samples x n_features)
            y: Target vector (n_samples,)
            feature_names: List of feature names
            
        Returns:
            SymbolicRegressionResult with discovered expressions
        """
        logger.info("Starting symbolic regression...")
        
        # Prepare data
        X_array = X.values
        y_array = y.values
        
        if feature_names is None:
            feature_names = [f"x_{i}" for i in range(X.shape[1])]
        
        # Convert config to PySR parameters
        pysr_params = config_to_dict(self.config)
        
        # Set up PySR
        pysr_params["prog_use_numpy"] = True
        
        # Run PySR
        logger.info(f"Running PySR with {len(feature_names)} features...")
        logger.info(f"Configuration: max_size={self.config.max_size}, "
                   f"parsimony={self.config.parsimony}, "
                   f"n_populations={self.config.n_populations}")
        
        try:
            # Run PySR
            model = pysr.PySRRegressor(
                niterations=self.config.max_iterations,
                maxsize=self.config.max_size,
                maxdepth=self.config.max_depth,
                binary_operators=self.config.binary_operators,
                unary_operators=self.config.unary_operators,
                parsimony=self.config.parsimony,
                pop_size=self.config.population_size,
                n_populations=self.config.n_populations,
                maxtime=self.config.max_time,
                loss_function=self.config.loss_function,
                deterministic=self.config.deterministic,
                random_state=self.config.random_seed,
                early_stop_condition="loss" if self.config.early_stopping else None,
                early_stop_patience=self.config.early_stopping_patience,
                complexity_of_operators=self.config.complexity_penalties,
                prog_use_numpy=True,
                verbosity=0
            )
            
            # Fit the model
            model.fit(X_array, y_array, variable_names=feature_names)
            
            # Extract results
            result = self._extract_results(model, feature_names)
            
            logger.info(f"Discovered {len(result.expressions)} expressions")
            if result.best_expression:
                logger.info(f"Best expression: {result.best_expression.expression}")
                logger.info(f"Best loss: {result.best_expression.loss:.6f}")
                logger.info(f"Best complexity: {result.best_expression.complexity}")
            
            return result
            
        except Exception as e:
            logger.error(f"PySR failed: {str(e)}")
            raise
    
    def _extract_results(self, 
                        model,
                        feature_names: List[str]) -> SymbolicRegressionResult:
        """Extract results from PySR model."""
        expressions = []
        
        # Get all expressions from the model
        if hasattr(model, 'results_'):
            results = model.results_
        elif hasattr(model, 'equation_'):
            results = [model.equation_]
        else:
            results = []
        
        # Process each expression
        for i, eq in enumerate(results):
            try:
                # Get expression string
                if hasattr(eq, 'expression'):
                    expr_str = eq.expression
                elif hasattr(eq, '__str__'):
                    expr_str = str(eq)
                else:
                    continue
                
                # Get loss
                if hasattr(eq, 'loss'):
                    loss = eq.loss
                elif hasattr(eq, 'score'):
                    loss = -eq.score  # Convert score to loss
                else:
                    loss = np.inf
                
                # Get complexity
                if hasattr(eq, 'complexity'):
                    complexity = eq.complexity
                else:
                    complexity = len(expr_str)
                
                # Create AlphaExpression
                alpha_expr = AlphaExpression(
                    expression=expr_str,
                    loss=loss,
                    complexity=complexity,
                    feature_names=feature_names,
                    rank=i + 1
                )
                
                expressions.append(alpha_expr)
                
            except Exception as e:
                logger.warning(f"Failed to process expression {i}: {str(e)}")
        
        # Sort expressions by loss (ascending)
        expressions.sort(key=lambda x: x.loss)
        
        # Get best expression
        best_expression = expressions[0] if expressions else None
        
        # Create result
        result = SymbolicRegressionResult(
            expressions=expressions[:self.config.n_top_expressions],
            best_expression=best_expression,
            config=self.config
        )
        
        return result
    
    def predict(self, 
               expression: Union[str, AlphaExpression],
               X: pd.DataFrame) -> pd.Series:
        """
        Predict using a discovered expression.
        
        Args:
            expression: Expression string or AlphaExpression
            X: Feature matrix
            
        Returns:
            Predicted values
        """
        if isinstance(expression, AlphaExpression):
            expr_str = expression.expression
        else:
            expr_str = expression
        
        # Create a simple evaluator
        feature_names = X.columns.tolist()
        
        # Create a lambda function for the expression
        try:
            # Replace feature names with array indices
            for i, name in enumerate(feature_names):
                expr_str = expr_str.replace(name, f"X[:, {i}]")
            
            # Create evaluation function
            def evaluate(X):
                X = np.array(X)
                return eval(expr_str)
            
            # Evaluate
            predictions = evaluate(X)
            
            return pd.Series(predictions, index=X.index)
            
        except Exception as e:
            logger.error(f"Failed to evaluate expression '{expression}': {str(e)}")
            raise


class SymbolicRegressionPipeline:
    """
    Pipeline for running symbolic regression on financial data.
    
    Manages the complete workflow from feature preparation to alpha discovery.
    """
    
    def __init__(self, 
                 config: Optional[PySRConfig] = None):
        """
        Initialize the pipeline.
        
        Args:
            config: PySR configuration
        """
        self.regressor = SymbolicRegressor(config)
        self.config = config or get_default_config()
    
    def run(self, 
            features: pd.DataFrame,
            target: pd.Series,
            feature_names: Optional[List[str]] = None) -> SymbolicRegressionResult:
        """
        Run symbolic regression on features and target.
        
        Args:
            features: Feature matrix
            target: Target vector (forward returns)
            feature_names: List of feature names
            
        Returns:
            SymbolicRegressionResult with discovered alphas
        """
        logger.info("Running symbolic regression pipeline...")
        
        # Align features and target
        features, target = self._align_data(features, target)
        
        # Remove rows with NaN values
        features = features.dropna()
        target = target.loc[features.index]
        
        logger.info(f"Training on {len(features)} samples with {len(features.columns)} features")
        
        # Run symbolic regression
        result = self.regressor.fit(features, target, feature_names)
        
        return result
    
    def _align_data(self, 
                    features: pd.DataFrame,
                    target: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Align features and target, removing mismatched indices."""
        # Find common index
        common_index = features.index.intersection(target.index)
        
        if len(common_index) == 0:
            raise ValueError("No common indices between features and target")
        
        return features.loc[common_index], target.loc[common_index]


def run_symbolic_regression(
    features: pd.DataFrame,
    target: pd.Series,
    config: Optional[PySRConfig] = None,
    feature_names: Optional[List[str]] = None
) -> SymbolicRegressionResult:
    """
    Convenience function to run symbolic regression.
    
    Args:
        features: Feature matrix
        target: Target vector
        config: PySR configuration
        feature_names: List of feature names
        
    Returns:
        SymbolicRegressionResult with discovered alphas
    """
    pipeline = SymbolicRegressionPipeline(config)
    return pipeline.run(features, target, feature_names)


def discover_alphas(
    features: pd.DataFrame,
    target: pd.Series,
    config: Optional[PySRConfig] = None,
    n_alphas: int = 10,
    feature_names: Optional[List[str]] = None
) -> List[AlphaExpression]:
    """
    Discover multiple alpha expressions.
    
    Args:
        features: Feature matrix
        target: Target vector
        config: PySR configuration
        n_alphas: Number of alphas to return
        feature_names: List of feature names
        
    Returns:
        List of discovered AlphaExpression objects
    """
    result = run_symbolic_regression(features, target, config, feature_names)
    
    # Return top n alphas
    return result.expressions[:n_alphas]
