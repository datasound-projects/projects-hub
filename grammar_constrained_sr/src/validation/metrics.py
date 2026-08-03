"""
Performance metrics for alpha evaluation.

This module implements various performance metrics for evaluating
alpha expressions and trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
import logging
from scipy.stats import spearmanr, pearsonr

logger = logging.getLogger(__name__)


@dataclass
class InformationCoefficient:
    """
    Information Coefficient (IC) metric.
    
    Measures the correlation between alpha predictions and actual forward returns.
    """
    
    # IC value
    value: float
    
    # P-value
    p_value: float
    
    # Number of observations
    n_observations: int
    
    # Method used ('spearman' or 'pearson')
    method: str = "spearman"
    
    @classmethod
    def compute(cls, 
                predictions: pd.Series,
                actuals: pd.Series,
                method: str = "spearman") -> 'InformationCoefficient':
        """
        Compute Information Coefficient.
        
        Args:
            predictions: Alpha predictions
            actuals: Actual forward returns
            method: Correlation method ('spearman' or 'pearson')
            
        Returns:
            InformationCoefficient object
        """
        # Align indices
        predictions, actuals = cls._align_series(predictions, actuals)
        
        if len(predictions) < 2 or len(actuals) < 2:
            return cls(value=0.0, p_value=1.0, n_observations=len(predictions), method=method)
        
        try:
            if method == "spearman":
                corr, p_value = spearmanr(predictions, actuals)
            else:  # pearson
                corr, p_value = pearsonr(predictions, actuals)
            
            return cls(
                value=float(corr),
                p_value=float(p_value),
                n_observations=len(predictions),
                method=method
            )
        except Exception as e:
            logger.warning(f"Failed to compute {method} correlation: {str(e)}")
            return cls(value=0.0, p_value=1.0, n_observations=len(predictions), method=method)
    
    @staticmethod
    def _align_series(series1: pd.Series, series2: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Align two series by index."""
        common_index = series1.index.intersection(series2.index)
        return series1.loc[common_index], series2.loc[common_index]


@dataclass
class PerformanceMetrics:
    """
    Comprehensive performance metrics for an alpha.
    """
    
    # Information Coefficient
    ic: float
    ic_p_value: float
    
    # IC stability (fraction of positive daily ICs)
    ic_stability: float
    
    # Sharpe ratio
    sharpe_ratio: float
    
    # Annualized return
    annualized_return: float
    
    # Annualized volatility
    annualized_volatility: float
    
    # Maximum drawdown
    max_drawdown: float
    
    # Win rate
    win_rate: float
    
    # Turnover
    turnover: float
    
    # Number of observations
    n_observations: int
    
    # Complexity
    complexity: int
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'ic': self.ic,
            'ic_p_value': self.ic_p_value,
            'ic_stability': self.ic_stability,
            'sharpe_ratio': self.sharpe_ratio,
            'annualized_return': self.annualized_return,
            'annualized_volatility': self.annualized_volatility,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'turnover': self.turnover,
            'n_observations': self.n_observations,
            'complexity': self.complexity
        }


class MetricsCalculator:
    """
    Calculator for various performance metrics.
    """
    
    def __init__(self, 
                 n_quintiles: int = 5,
                 trading_days: int = 252):
        """
        Initialize the calculator.
        
        Args:
            n_quintiles: Number of quintiles for long-short portfolio
            trading_days: Number of trading days per year
        """
        self.n_quintiles = n_quintiles
        self.trading_days = trading_days
    
    def compute_all_metrics(self, 
                          alpha: pd.Series,
                          target: pd.Series) -> PerformanceMetrics:
        """
        Compute all performance metrics for an alpha.
        
        Args:
            alpha: Alpha signal values
            target: Forward return values
            
        Returns:
            PerformanceMetrics object
        """
        # Compute IC
        ic_result = InformationCoefficient.compute(alpha, target, method="spearman")
        
        # Compute IC stability
        ic_stability = self.compute_ic_stability(alpha, target)
        
        # Compute Sharpe ratio
        sharpe_ratio = self.compute_sharpe_ratio(alpha, target)
        
        # Compute annualized return and volatility
        annualized_return, annualized_volatility = self.compute_annualized_stats(alpha, target)
        
        # Compute max drawdown
        max_drawdown = self.compute_max_drawdown(alpha, target)
        
        # Compute win rate
        win_rate = self.compute_win_rate(alpha, target)
        
        # Compute turnover
        turnover = self.compute_turnover(alpha)
        
        return PerformanceMetrics(
            ic=ic_result.value,
            ic_p_value=ic_result.p_value,
            ic_stability=ic_stability,
            sharpe_ratio=sharpe_ratio,
            annualized_return=annualized_return,
            annualized_volatility=annualized_volatility,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            turnover=turnover,
            n_observations=ic_result.n_observations,
            complexity=0  # Will be set by caller
        )
    
    def compute_ic_stability(self, 
                           alpha: pd.Series,
                           target: pd.Series) -> float:
        """
        Compute IC stability.
        
        Args:
            alpha: Alpha signal values
            target: Forward return values
            
        Returns:
            IC stability (fraction of positive daily ICs)
        """
        # Align indices
        alpha, target = InformationCoefficient._align_series(alpha, target)
        
        if len(alpha) == 0 or len(target) == 0:
            return 0.0
        
        # For panel data with MultiIndex
        if isinstance(alpha.index, pd.MultiIndex):
            dates = alpha.index.get_level_values(0).unique()
            
            positive_count = 0
            total_count = 0
            
            for date in dates:
                alpha_date = alpha.loc[date]
                target_date = target.loc[date]
                
                if len(alpha_date) > 1 and len(target_date) > 1:
                    try:
                        corr, _ = spearmanr(alpha_date, target_date)
                        if not np.isnan(corr):
                            total_count += 1
                            if corr > 0:
                                positive_count += 1
                    except:
                        pass
            
            if total_count == 0:
                return 0.0
            
            return positive_count / total_count
        else:
            # For simple series
            ic = InformationCoefficient.compute(alpha, target).value
            return 1.0 if ic > 0 else 0.0
    
    def compute_sharpe_ratio(self, 
                            alpha: pd.Series,
                            target: pd.Series) -> float:
        """
        Compute annualized Sharpe ratio of a quintile long-short portfolio.
        
        Args:
            alpha: Alpha signal values
            target: Forward return values
            
        Returns:
            Annualized Sharpe ratio
        """
        # Align indices
        alpha, target = InformationCoefficient._align_series(alpha, target)
        
        if len(alpha) == 0 or len(target) == 0:
            return 0.0
        
        # For panel data
        if isinstance(alpha.index, pd.MultiIndex):
            dates = alpha.index.get_level_values(0).unique()
            
            portfolio_returns = []
            
            for date in dates:
                alpha_date = alpha.loc[date]
                target_date = target.loc[date]
                
                if len(alpha_date) >= self.n_quintiles:
                    # Sort by alpha
                    sorted_alpha = alpha_date.sort_values()
                    sorted_target = target_date.loc[sorted_alpha.index]
                    
                    # Create long-short portfolio
                    n_assets = len(alpha_date)
                    n_long = n_assets // self.n_quintiles
                    n_short = n_assets // self.n_quintiles
                    
                    long_returns = sorted_target.iloc[-n_long:].mean()
                    short_returns = sorted_target.iloc[:n_short].mean()
                    
                    portfolio_return = long_returns - short_returns
                    portfolio_returns.append(portfolio_return)
            
            if len(portfolio_returns) < 2:
                return 0.0
            
            # Compute Sharpe ratio
            portfolio_returns = pd.Series(portfolio_returns)
            mean_return = portfolio_returns.mean()
            std_return = portfolio_returns.std()
            
            if std_return <= 0:
                return 0.0
            
            # Annualize
            annualized_mean = mean_return * self.trading_days
            annualized_std = std_return * np.sqrt(self.trading_days)
            
            sharpe = annualized_mean / annualized_std
            
            return float(sharpe)
        else:
            # For simple series, approximate from IC
            ic = InformationCoefficient.compute(alpha, target).value
            return ic * np.sqrt(self.trading_days)
    
    def compute_annualized_stats(self, 
                                alpha: pd.Series,
                                target: pd.Series) -> Tuple[float, float]:
        """
        Compute annualized return and volatility.
        
        Args:
            alpha: Alpha signal values
            target: Forward return values
            
        Returns:
            Tuple of (annualized_return, annualized_volatility)
        """
        # For panel data, compute portfolio returns
        if isinstance(alpha.index, pd.MultiIndex):
            dates = alpha.index.get_level_values(0).unique()
            
            portfolio_returns = []
            
            for date in dates:
                alpha_date = alpha.loc[date]
                target_date = target.loc[date]
                
                if len(alpha_date) >= self.n_quintiles:
                    sorted_alpha = alpha_date.sort_values()
                    sorted_target = target_date.loc[sorted_alpha.index]
                    
                    n_assets = len(alpha_date)
                    n_long = n_assets // self.n_quintiles
                    n_short = n_assets // self.n_quintiles
                    
                    long_returns = sorted_target.iloc[-n_long:].mean()
                    short_returns = sorted_target.iloc[:n_short].mean()
                    
                    portfolio_return = long_returns - short_returns
                    portfolio_returns.append(portfolio_return)
            
            if len(portfolio_returns) < 2:
                return 0.0, 0.0
            
            portfolio_returns = pd.Series(portfolio_returns)
            mean_return = portfolio_returns.mean()
            std_return = portfolio_returns.std()
            
            annualized_return = mean_return * self.trading_days
            annualized_volatility = std_return * np.sqrt(self.trading_days)
            
            return annualized_return, annualized_volatility
        else:
            # For simple series
            mean_return = target.mean()
            std_return = target.std()
            
            annualized_return = mean_return * self.trading_days
            annualized_volatility = std_return * np.sqrt(self.trading_days)
            
            return annualized_return, annualized_volatility
    
    def compute_max_drawdown(self, 
                             alpha: pd.Series,
                             target: pd.Series) -> float:
        """
        Compute maximum drawdown.
        
        Args:
            alpha: Alpha signal values
            target: Forward return values
            
        Returns:
            Maximum drawdown
        """
        # For panel data, compute portfolio cumulative returns
        if isinstance(alpha.index, pd.MultiIndex):
            dates = alpha.index.get_level_values(0).unique()
            
            portfolio_returns = []
            
            for date in dates:
                alpha_date = alpha.loc[date]
                target_date = target.loc[date]
                
                if len(alpha_date) >= self.n_quintiles:
                    sorted_alpha = alpha_date.sort_values()
                    sorted_target = target_date.loc[sorted_alpha.index]
                    
                    n_assets = len(alpha_date)
                    n_long = n_assets // self.n_quintiles
                    n_short = n_assets // self.n_quintiles
                    
                    long_returns = sorted_target.iloc[-n_long:].mean()
                    short_returns = sorted_target.iloc[:n_short].mean()
                    
                    portfolio_return = long_returns - short_returns
                    portfolio_returns.append(portfolio_return)
            
            if len(portfolio_returns) < 2:
                return 0.0
            
            # Compute cumulative returns
            cumulative_returns = (1 + pd.Series(portfolio_returns)).cumprod()
            
            # Compute drawdowns
            running_max = cumulative_returns.cummax()
            drawdowns = (cumulative_returns - running_max) / running_max
            
            max_drawdown = drawdowns.min()
            
            return float(max_drawdown)
        else:
            # For simple series
            cumulative_returns = (1 + target).cumprod()
            running_max = cumulative_returns.cummax()
            drawdowns = (cumulative_returns - running_max) / running_max
            
            return float(drawdowns.min())
    
    def compute_win_rate(self, 
                         alpha: pd.Series,
                         target: pd.Series) -> float:
        """
        Compute win rate.
        
        Args:
            alpha: Alpha signal values
            target: Forward return values
            
        Returns:
            Win rate (fraction of positive returns)
        """
        # Align indices
        alpha, target = InformationCoefficient._align_series(alpha, target)
        
        if len(target) == 0:
            return 0.0
        
        # For panel data, compute portfolio win rate
        if isinstance(alpha.index, pd.MultiIndex):
            dates = alpha.index.get_level_values(0).unique()
            
            wins = 0
            total = 0
            
            for date in dates:
                alpha_date = alpha.loc[date]
                target_date = target.loc[date]
                
                if len(alpha_date) >= self.n_quintiles:
                    sorted_alpha = alpha_date.sort_values()
                    sorted_target = target_date.loc[sorted_alpha.index]
                    
                    n_assets = len(alpha_date)
                    n_long = n_assets // self.n_quintiles
                    n_short = n_assets // self.n_quintiles
                    
                    long_returns = sorted_target.iloc[-n_long:].mean()
                    short_returns = sorted_target.iloc[:n_short].mean()
                    
                    portfolio_return = long_returns - short_returns
                    
                    total += 1
                    if portfolio_return > 0:
                        wins += 1
            
            if total == 0:
                return 0.0
            
            return wins / total
        else:
            # For simple series
            return float((target > 0).mean())
    
    def compute_turnover(self, 
                         alpha: pd.Series) -> float:
        """
        Compute daily turnover.
        
        Args:
            alpha: Alpha signal values
            
        Returns:
            Daily turnover
        """
        if not isinstance(alpha.index, pd.MultiIndex):
            return 0.0
        
        dates = alpha.index.get_level_values(0).unique()
        
        if len(dates) < 2:
            return 0.0
        
        turnovers = []
        
        for i in range(1, len(dates)):
            alpha_prev = alpha.loc[dates[i-1]]
            alpha_curr = alpha.loc[dates[i]]
            
            if len(alpha_prev) == 0 or len(alpha_curr) == 0:
                continue
            
            # Get top and bottom quintile
            n_assets = len(alpha_prev)
            n_quintile = n_assets // self.n_quintiles
            
            prev_long = alpha_prev.nlargest(n_quintile).index
            prev_short = alpha_prev.nsmallest(n_quintile).index
            
            curr_long = alpha_curr.nlargest(n_quintile).index
            curr_short = alpha_curr.nsmallest(n_quintile).index
            
            # Compute turnover
            long_turnover = len(set(prev_long) - set(curr_long)) / n_quintile
            short_turnover = len(set(prev_short) - set(curr_short)) / n_quintile
            
            total_turnover = (long_turnover + short_turnover) / 2
            turnovers.append(total_turnover)
        
        if len(turnovers) == 0:
            return 0.0
        
        return float(np.mean(turnovers))


def compute_all_metrics(alpha: pd.Series, 
                       target: pd.Series,
                       n_quintiles: int = 5,
                       trading_days: int = 252) -> PerformanceMetrics:
    """
    Convenience function to compute all metrics.
    
    Args:
        alpha: Alpha signal values
        target: Forward return values
        n_quintiles: Number of quintiles
        trading_days: Number of trading days per year
        
    Returns:
        PerformanceMetrics object
    """
    calculator = MetricsCalculator(n_quintiles, trading_days)
    return calculator.compute_all_metrics(alpha, target)


def compute_ic_stability(alpha: pd.Series, 
                         target: pd.Series) -> float:
    """
    Convenience function to compute IC stability.
    
    Args:
        alpha: Alpha signal values
        target: Forward return values
        
    Returns:
        IC stability
    """
    calculator = MetricsCalculator()
    return calculator.compute_ic_stability(alpha, target)
