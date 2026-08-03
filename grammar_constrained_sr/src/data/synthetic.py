"""
Synthetic data generation module.

This module provides functionality to generate synthetic financial time series
for testing and validation of the alpha discovery pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""
    num_assets: int = 50
    num_days: int = 1000
    start_date: str = "2020-01-01"
    base_volatility: float = 0.02
    volatility_cluster_strength: float = 0.5
    leverage_effect_strength: float = 0.3
    mean_reversion_strength: float = 0.1
    momentum_strength: float = 0.05
    fat_tail_alpha: float = 1.5
    market_factor_strength: float = 0.5
    idiosyncratic_strength: float = 0.5
    seed: Optional[int] = 42
    

@dataclass
class PlantedAlpha:
    """Configuration for a planted alpha signal."""
    expression: str
    signal_strength: float = 0.1
    active_fraction: float = 0.5
    window: int = 20
    

class SyntheticDataGenerator:
    """
    Generates synthetic financial time series with realistic stylized facts.
    
    Implements various stochastic processes to simulate:
    - Fat tails (via stable distributions)
    - Volatility clustering (via GARCH-like processes)
    - Leverage effect (asymmetric volatility)
    - Mean reversion and momentum
    - Market factor structure
    """
    
    def __init__(self, 
                 config: Optional[SyntheticConfig] = None,
                 planted_alphas: Optional[List[PlantedAlpha]] = None):
        """
        Initialize the SyntheticDataGenerator.
        
        Args:
            config: Configuration for data generation
            planted_alphas: List of alpha signals to plant in the data
        """
        self.config = config or SyntheticConfig()
        self.planted_alphas = planted_alphas or []
        self.rng = np.random.default_rng(self.config.seed)
        
    def generate_dates(self) -> pd.DatetimeIndex:
        """Generate trading dates."""
        start = datetime.strptime(self.config.start_date, "%Y-%m-%d")
        dates = pd.date_range(
            start=start,
            periods=self.config.num_days,
            freq='B'  # Business days
        )
        return dates
    
    def generate_market_factor(self, 
                              dates: pd.DatetimeIndex) -> pd.Series:
        """
        Generate a market factor time series with stylized facts.
        
        Args:
            dates: DatetimeIndex of dates
            
        Returns:
            Series with market factor returns
        """
        n = len(dates)
        
        # Generate volatility process (volatility clustering)
        sigma = self._generate_volatility_process(n)
        
        # Generate innovations with fat tails
        innovations = self._generate_fat_tailed_innovations(n)
        
        # Apply leverage effect (negative returns increase volatility)
        returns, sigma = self._apply_leverage_effect(innovations, sigma)
        
        # Apply mean reversion
        returns = self._apply_mean_reversion(returns)
        
        # Apply momentum
        returns = self._apply_momentum(returns)
        
        return pd.Series(returns, index=dates)
    
    def _generate_volatility_process(self, 
                                     n: int) -> np.ndarray:
        """Generate a volatility process with clustering."""
        # Start with constant volatility
        sigma = np.full(n, self.config.base_volatility)
        
        # Add volatility clustering using a simple AR process
        for i in range(1, n):
            # Volatility shock
            shock = self.rng.normal(0, 0.01)
            # AR(1) process for volatility
            sigma[i] = 0.95 * sigma[i-1] + 0.05 * self.config.base_volatility + shock
            # Ensure positive
            sigma[i] = max(sigma[i], 0.001)
        
        return sigma
    
    def _generate_fat_tailed_innovations(self, 
                                        n: int) -> np.ndarray:
        """Generate innovations with fat tails using stable distribution."""
        # Use a mixture of normal and stable distribution for fat tails
        alpha = self.config.fat_tail_alpha
        
        # Generate stable distribution innovations
        # For simplicity, we use a t-distribution approximation
        df = 2 * alpha  # Degrees of freedom (lower = fatter tails)
        innovations = self.rng.standard_t(df, size=n)
        
        return innovations
    
    def _apply_leverage_effect(self, 
                              innovations: np.ndarray,
                              sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply leverage effect to returns and volatility."""
        n = len(innovations)
        returns = np.zeros(n)
        
        for i in range(n):
            # Scale innovation by volatility
            returns[i] = sigma[i] * innovations[i]
            
            # Leverage effect: negative returns increase future volatility
            if i < n - 1 and returns[i] < 0:
                sigma[i+1] *= (1 + self.config.leverage_effect_strength * abs(returns[i]))
        
        return returns, sigma
    
    def _apply_mean_reversion(self, 
                             returns: np.ndarray) -> np.ndarray:
        """Apply mean reversion to returns."""
        n = len(returns)
        
        for i in range(1, n):
            # Simple mean reversion: if yesterday was positive, today is more likely negative
            if i > 0:
                returns[i] -= self.config.mean_reversion_strength * returns[i-1]
        
        return returns
    
    def _apply_momentum(self, 
                       returns: np.ndarray) -> np.ndarray:
        """Apply momentum effect to returns."""
        n = len(returns)
        
        # Compute cumulative returns
        cumulative = np.cumsum(returns)
        
        for i in range(20, n):  # Start after 20 days
            # Momentum: assets that have done well tend to continue doing well
            past_return = cumulative[i] - cumulative[i-20]
            returns[i] += self.config.momentum_strength * past_return
        
        return returns
    
    def generate_idiosyncratic_returns(self, 
                                       n: int,
                                       dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Generate idiosyncratic returns for all assets."""
        # Generate independent idiosyncratic shocks
        idio_returns = pd.DataFrame(
            self.rng.normal(0, self.config.idiosyncratic_strength, size=(n, self.config.num_assets)),
            index=dates
        )
        
        return idio_returns
    
    def generate_asset_returns(self, 
                              market_factor: pd.Series,
                              dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Generate returns for all assets using a factor model.
        
        Args:
            market_factor: Market factor returns
            dates: DatetimeIndex of dates
            
        Returns:
            DataFrame with returns for all assets
        """
        n = len(dates)
        
        # Generate factor loadings (betas) for each asset
        betas = self.rng.uniform(0.5, 1.5, size=self.config.num_assets)
        
        # Generate idiosyncratic returns
        idio_returns = self.generate_idiosyncratic_returns(n, dates)
        
        # Combine market factor and idiosyncratic returns
        asset_returns = pd.DataFrame(
            np.outer(market_factor.values, betas) + idio_returns.values,
            index=dates
        )
        
        return asset_returns
    
    def generate_ohlcv_data(self, 
                           returns: pd.DataFrame,
                           dates: pd.DatetimeIndex) -> Dict[str, pd.DataFrame]:
        """
        Convert returns to OHLCV data.
        
        Args:
            returns: DataFrame with returns for all assets
            dates: DatetimeIndex of dates
            
        Returns:
            Dictionary mapping symbol to OHLCV DataFrame
        """
        assets_data = {}
        
        for i, symbol in enumerate([f"ASSET_{i:03d}" for i in range(self.config.num_assets)]):
            # Start with price = 100
            prices = np.cumprod(1 + returns.iloc[:, i].values)
            prices = 100 * prices / prices[0]  # Scale to start at 100
            
            # Generate OHLC data (simplified: open = previous close, high/low = close +/- random)
            opens = np.roll(prices, 1)
            opens[0] = prices[0]
            
            # Add some randomness to high/low
            spread = abs(returns.iloc[:, i].values) * 0.5
            highs = prices + spread
            lows = prices - spread
            
            # Generate volume (correlated with absolute returns)
            abs_returns = np.abs(returns.iloc[:, i].values)
            volumes = 1000000 + 500000 * abs_returns * 100  # Scale up for realistic volume
            
            df = pd.DataFrame({
                'open': opens,
                'high': highs,
                'low': lows,
                'close': prices,
                'volume': volumes
            }, index=dates)
            
            assets_data[symbol] = df
        
        return assets_data
    
    def plant_alpha_signal(self, 
                          returns: pd.DataFrame,
                          alpha: PlantedAlpha) -> pd.DataFrame:
        """
        Plant an alpha signal in the returns.
        
        Args:
            returns: DataFrame with returns
            alpha: PlantedAlpha configuration
            
        Returns:
            DataFrame with alpha signal planted
        """
        n, m = returns.shape
        dates = returns.index
        
        # Compute the alpha signal
        if alpha.expression == "mean_reversion":
            # Simple mean reversion: buy after negative returns, sell after positive
            for i in range(alpha.window, n):
                past_return = returns.iloc[i-alpha.window:i, :].mean(axis=0)
                # Add signal: assets with negative past returns tend to have positive future returns
                for j in range(m):
                    if self.rng.random() < alpha.active_fraction:
                        returns.iloc[i, j] += alpha.signal_strength * (-past_return[j])
        
        elif alpha.expression == "momentum":
            # Momentum: assets with positive past returns tend to have positive future returns
            for i in range(alpha.window, n):
                past_return = returns.iloc[i-alpha.window:i, :].sum(axis=0)
                for j in range(m):
                    if self.rng.random() < alpha.active_fraction:
                        returns.iloc[i, j] += alpha.signal_strength * past_return[j]
        
        elif alpha.expression == "volatility":
            # Low volatility predicts higher returns (volatility anomaly)
            for i in range(alpha.window, n):
                past_volatility = returns.iloc[i-alpha.window:i, :].std(axis=0)
                for j in range(m):
                    if self.rng.random() < alpha.active_fraction:
                        returns.iloc[i, j] += alpha.signal_strength * (-past_volatility[j])
        
        return returns
    
    def generate(self) -> Dict[str, pd.DataFrame]:
        """
        Generate complete synthetic dataset.
        
        Returns:
            Dictionary mapping symbol to OHLCV DataFrame
        """
        logger.info(f"Generating synthetic data: {self.config.num_assets} assets, {self.config.num_days} days")
        
        # Generate dates
        dates = self.generate_dates()
        
        # Generate market factor
        market_factor = self.generate_market_factor(dates)
        
        # Generate asset returns
        asset_returns = self.generate_asset_returns(market_factor, dates)
        
        # Plant alpha signals
        for alpha in self.planted_alphas:
            asset_returns = self.plant_alpha_signal(asset_returns, alpha)
            logger.info(f"Planted alpha: {alpha.expression} (strength={alpha.signal_strength})")
        
        # Convert to OHLCV
        ohlcv_data = self.generate_ohlcv_data(asset_returns, dates)
        
        logger.info(f"Generated {len(ohlcv_data)} assets with OHLCV data")
        
        return ohlcv_data


def generate_synthetic_panel(num_assets: int = 50,
                            num_days: int = 1000,
                            start_date: str = "2020-01-01",
                            planted_alpha: Optional[str] = None,
                            signal_strength: float = 0.1,
                            seed: Optional[int] = 42) -> pd.DataFrame:
    """
    Convenience function to generate a synthetic panel with optional planted alpha.
    
    Args:
        num_assets: Number of assets
        num_days: Number of trading days
        start_date: Start date
        planted_alpha: Type of alpha to plant ('mean_reversion', 'momentum', 'volatility')
        signal_strength: Strength of the planted signal
        seed: Random seed
        
    Returns:
        Panel DataFrame with MultiIndex (symbol, date)
    """
    from .loader import DataLoader
    from .panel import PanelConstructor, PanelConfig
    
    # Configure planted alpha
    planted_alphas = []
    if planted_alpha:
        planted_alphas = [
            PlantedAlpha(
                expression=planted_alpha,
                signal_strength=signal_strength,
                window=20
            )
        ]
    
    # Generate synthetic data
    config = SyntheticConfig(
        num_assets=num_assets,
        num_days=num_days,
        start_date=start_date,
        seed=seed
    )
    
    generator = SyntheticDataGenerator(config, planted_alphas)
    asset_data = generator.generate()
    
    # Convert to panel
    loader = DataLoader()
    preprocessed = {}
    
    for symbol, df in asset_data.items():
        preprocessed[symbol] = loader.preprocess_asset(
            loader.assets[symbol],
            forward_return_horizon=1
        )
    
    constructor = PanelConstructor(PanelConfig(forward_return_horizon=1))
    panel = constructor.construct_from_dict(preprocessed)
    panel = constructor.handle_missing_data(panel)
    
    return panel
