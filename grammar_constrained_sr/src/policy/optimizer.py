"""
Policy optimizer for finding optimal execution rules.

This module implements optimization of policy functions to maximize
the growth rate of the equity curve.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass, field
import logging

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna not available. Policy optimization will not work.")

from .base import Policy, PolicyConfig, ParameterizedPolicy
from .quintile import QuintileLongShortPolicy
from .threshold import ThresholdPolicy, ZScorePolicy
from .kelly import KellyCriterionPolicy

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for policy optimization."""
    # Objective function: 'growth_rate', 'sharpe', 'return', 'win_rate'
    objective: str = "growth_rate"
    
    # Number of trials
    n_trials: int = 100
    
    # Timeout in seconds
    timeout: int = 300
    
    # Whether to use pruning
    prune: bool = True
    
    # Random seed
    random_seed: Optional[int] = None


@dataclass
class OptimizationResult:
    """Result of policy optimization."""
    best_policy: Policy
    best_parameters: Dict[str, Any]
    best_value: float
    history: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'best_parameters': self.best_parameters,
            'best_value': self.best_value,
            'history': self.history
        }


class PolicyOptimizer:
    """
    Optimizer for policy functions.
    
    Uses Bayesian optimization (via Optuna) to find the policy parameters
    that maximize the specified objective.
    """
    
    def __init__(self, 
                 config: Optional[OptimizationConfig] = None):
        """
        Initialize the PolicyOptimizer.
        
        Args:
            config: Optimization configuration
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for policy optimization. Install with: pip install optuna")
        
        self.config = config or OptimizationConfig()
    
    def optimize(self, 
                policy: ParameterizedPolicy,
                alpha: pd.Series,
                returns: pd.Series,
                dates: Optional[pd.DatetimeIndex] = None) -> OptimizationResult:
        """
        Optimize a policy function.
        
        Args:
            policy: Parameterized policy to optimize
            alpha: Alpha signal values
            returns: Forward return values
            dates: Optional datetime index
            
        Returns:
            OptimizationResult with best policy and parameters
        """
        logger.info(f"Optimizing {policy.config.name} policy...")
        
        # Create Optuna study
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.config.random_seed),
            pruner=optuna.pruners.MedianPruner() if self.config.prune else None
        )
        
        # Define objective function
        def objective(trial: optuna.Trial) -> float:
            # Suggest parameters
            params = self._suggest_parameters(trial, policy)
            
            # Set parameters on policy
            policy.set_parameters(**params)
            
            # Compute portfolio weights
            weights = policy.compute_weights(alpha)
            
            # Compute portfolio returns
            portfolio_returns = (weights * returns).sum()
            
            # Compute objective value
            value = self._compute_objective(portfolio_returns, dates)
            
            return value
        
        # Run optimization
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout
        )
        
        # Get best parameters
        best_params = study.best_params
        policy.set_parameters(**best_params)
        
        # Compute best value
        best_weights = policy.compute_weights(alpha)
        best_returns = (best_weights * returns).sum()
        best_value = self._compute_objective(best_returns, dates)
        
        # Create result
        result = OptimizationResult(
            best_policy=policy,
            best_parameters=best_params,
            best_value=best_value,
            history=[{'trial': t.number, 'value': t.value} for t in study.trials]
        )
        
        logger.info(f"Optimization complete. Best value: {best_value:.6f}")
        logger.info(f"Best parameters: {best_params}")
        
        return result
    
    def _suggest_parameters(self, 
                          trial: optuna.Trial,
                          policy: ParameterizedPolicy) -> Dict[str, Any]:
        """Suggest parameters for a trial."""
        params = {}
        
        # Get parameter grid for suggestions
        param_grid = policy.get_parameter_grid()
        if not param_grid:
            return params
        
        # Get parameter ranges from the grid
        param_ranges = self._get_parameter_ranges(param_grid)
        
        # Suggest each parameter
        for param_name, param_range in param_ranges.items():
            if isinstance(param_range, list) and len(param_range) > 0:
                # Categorical parameter
                params[param_name] = trial.suggest_categorical(param_name, param_range)
            elif isinstance(param_range, dict):
                # Numeric parameter with min/max
                if param_range.get('type') == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_range['min'],
                        param_range['max'],
                        step=param_range.get('step', 1)
                    )
                else:
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_range['min'],
                        param_range['max'],
                        step=param_range.get('step', 0.1)
                    )
            else:
                # Use default value
                params[param_name] = param_range
        
        return params
    
    def _get_parameter_ranges(self, 
                            param_grid: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract parameter ranges from a parameter grid."""
        param_ranges = {}
        
        for params in param_grid:
            for param_name, param_value in params.items():
                if param_name not in param_ranges:
                    param_ranges[param_name] = set()
                param_ranges[param_name].add(param_value)
        
        # Convert sets to lists or ranges
        for param_name, values in param_ranges.items():
            values = sorted(values)
            
            # Check if numeric
            if all(isinstance(v, (int, float)) for v in values):
                if all(isinstance(v, int) for v in values):
                    param_ranges[param_name] = {
                        'type': 'int',
                        'min': min(values),
                        'max': max(values),
                        'step': 1
                    }
                else:
                    param_ranges[param_name] = {
                        'type': 'float',
                        'min': min(values),
                        'max': max(values),
                        'step': 0.1
                    }
            else:
                # Categorical
                param_ranges[param_name] = list(values)
        
        return param_ranges
    
    def _compute_objective(self, 
                          portfolio_returns: pd.Series,
                          dates: Optional[pd.DatetimeIndex] = None) -> float:
        """Compute the objective function value."""
        if self.config.objective == "growth_rate":
            return self._compute_growth_rate(portfolio_returns)
        elif self.config.objective == "sharpe":
            return self._compute_sharpe_ratio(portfolio_returns)
        elif self.config.objective == "return":
            return float(portfolio_returns.mean())
        elif self.config.objective == "win_rate":
            return float((portfolio_returns > 0).mean())
        else:
            return self._compute_growth_rate(portfolio_returns)
    
    def _compute_growth_rate(self, 
                           returns: pd.Series) -> float:
        """Compute the growth rate (Kelly criterion)."""
        if len(returns) < 2:
            return 0.0
        
        # Growth rate = mean(log(1 + r)) * 252
        log_returns = np.log(1 + returns)
        growth_rate = log_returns.mean() * 252
        
        return float(growth_rate)
    
    def _compute_sharpe_ratio(self, 
                              returns: pd.Series) -> float:
        """Compute the Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        
        mean_return = returns.mean()
        std_return = returns.std()
        
        if std_return <= 0:
            return 0.0
        
        # Annualize
        annualized_mean = mean_return * 252
        annualized_std = std_return * np.sqrt(252)
        
        sharpe = annualized_mean / annualized_std
        
        return float(sharpe)
    
    def optimize_quintile_policy(self, 
                                alpha: pd.Series,
                                returns: pd.Series,
                                dates: Optional[pd.DatetimeIndex] = None) -> OptimizationResult:
        """
        Optimize a quintile long-short policy.
        
        Args:
            alpha: Alpha signal values
            returns: Forward return values
            dates: Optional datetime index
            
        Returns:
            OptimizationResult
        """
        policy = QuintileLongShortPolicy()
        return self.optimize(policy, alpha, returns, dates)
    
    def optimize_threshold_policy(self, 
                                  alpha: pd.Series,
                                  returns: pd.Series,
                                  dates: Optional[pd.DatetimeIndex] = None) -> OptimizationResult:
        """
        Optimize a threshold policy.
        
        Args:
            alpha: Alpha signal values
            returns: Forward return values
            dates: Optional datetime index
            
        Returns:
            OptimizationResult
        """
        policy = ThresholdPolicy()
        return self.optimize(policy, alpha, returns, dates)
    
    def optimize_zscore_policy(self, 
                               alpha: pd.Series,
                               returns: pd.Series,
                               dates: Optional[pd.DatetimeIndex] = None) -> OptimizationResult:
        """
        Optimize a z-score policy.
        
        Args:
            alpha: Alpha signal values
            returns: Forward return values
            dates: Optional datetime index
            
        Returns:
            OptimizationResult
        """
        policy = ZScorePolicy()
        return self.optimize(policy, alpha, returns, dates)
    
    def optimize_kelly_policy(self, 
                              alpha: pd.Series,
                              returns: pd.Series,
                              dates: Optional[pd.DatetimeIndex] = None) -> OptimizationResult:
        """
        Optimize a Kelly criterion policy.
        
        Args:
            alpha: Alpha signal values
            returns: Forward return values
            dates: Optional datetime index
            
        Returns:
            OptimizationResult
        """
        policy = KellyCriterionPolicy()
        return self.optimize(policy, alpha, returns, dates)


def optimize_policy(policy: ParameterizedPolicy,
                   alpha: pd.Series,
                   returns: pd.Series,
                   objective: str = "growth_rate",
                   n_trials: int = 100) -> OptimizationResult:
    """
    Convenience function to optimize a policy.
    
    Args:
        policy: Parameterized policy
        alpha: Alpha signal values
        returns: Forward return values
        objective: Objective function
        n_trials: Number of trials
        
    Returns:
        OptimizationResult
    """
    config = OptimizationConfig(
        objective=objective,
        n_trials=n_trials
    )
    
    optimizer = PolicyOptimizer(config)
    return optimizer.optimize(policy, alpha, returns)
