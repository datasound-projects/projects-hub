"""
Basic example demonstrating the core functionality of the alpha discovery pipeline.

This example shows:
1. Loading synthetic data
2. Computing stylized fact primitives
3. Running symbolic regression
4. Filtering candidates
5. Generating code
"""

import pandas as pd
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.data.synthetic import generate_synthetic_panel
from src.features.factory import FeatureFactory, FeatureConfig
from src.normalization import CombinedNormalizer, NormalizationConfig
from src.symbolic.regression import SymbolicRegressor
from src.symbolic.pysr_config import get_default_config
from src.validation.statistical import StatisticalFilter, FilterConfig
from src.codegen.generator import generate_alpha_code


def run_basic_example():
    """Run the basic example."""
    logger.info("Running basic example...")
    
    # Step 1: Generate synthetic data
    logger.info("Step 1: Generating synthetic data...")
    panel = generate_synthetic_panel(
        num_assets=30,
        num_days=500,
        start_date="2020-01-01",
        planted_alpha="mean_reversion",
        signal_strength=0.1,
        seed=42
    )
    
    logger.info(f"Generated panel with {len(panel.index.get_level_values('symbol').unique())} assets "
               f"and {len(panel.index.get_level_values('date').unique())} dates")
    
    # Step 2: Compute features
    logger.info("Step 2: Computing stylized fact primitives...")
    
    # Use a subset of features for faster execution
    feature_config = FeatureConfig(
        window_sizes=[5, 10, 20],
        include_wavelet=False,  # Disable for speed
        include_hilbert=True,
        include_spectral=True
    )
    
    feature_factory = FeatureFactory(feature_config)
    
    # Extract returns from panel
    returns = panel['log_return'].unstack(level='symbol')
    
    # Compute features for a few stylized facts
    features_list = []
    
    # Fat tails (Fact 1)
    fat_tails_features = feature_factory.compute_features_for_fact(1, panel)
    features_list.append(fat_tails_features)
    
    # Volatility clustering (Fact 2)
    vol_features = feature_factory.compute_features_for_fact(2, panel)
    features_list.append(vol_features)
    
    # Mean reversion (Fact 10)
    mean_reversion_features = feature_factory.compute_features_for_fact(10, panel)
    features_list.append(mean_reversion_features)
    
    # Combine all features
    all_features = pd.concat(features_list, axis=1)
    
    logger.info(f"Computed {len(all_features.columns)} features")
    
    # Step 3: Normalize features
    logger.info("Step 3: Normalizing features...")
    
    norm_config = NormalizationConfig(
        include_raw=False,
        include_rank=True,
        include_zscore=True
    )
    
    normalizer = CombinedNormalizer(norm_config)
    normalized_features = normalizer.normalize(all_features)
    
    logger.info(f"Normalized to {len(normalized_features.columns)} features")
    
    # Step 4: Prepare data for symbolic regression
    logger.info("Step 4: Preparing data for symbolic regression...")
    
    # Stack features to match panel structure
    features_stacked = normalized_features.stack()
    features_stacked.index = panel.index
    
    # Get target (forward returns)
    target = panel['forward_return']
    
    # Align features and target
    common_index = features_stacked.index.intersection(target.index)
    features_final = features_stacked.loc[common_index]
    target_final = target.loc[common_index]
    
    # Convert to DataFrame for PySR
    features_df = features_final.unstack(level='symbol')
    
    logger.info(f"Prepared {len(features_df)} samples with {len(features_df.columns)} features")
    
    # Step 5: Run symbolic regression
    logger.info("Step 5: Running symbolic regression...")
    
    # Use a smaller configuration for faster execution
    pysr_config = get_default_config()
    pysr_config.n_populations = 10
    pysr_config.population_size = 20
    pysr_config.max_iterations = 20
    pysr_config.max_time = 60
    
    regressor = SymbolicRegressor(pysr_config)
    
    # Get feature names
    feature_names = list(features_df.columns)
    
    # Run regression
    sr_result = regressor.fit(features_df, target_final, feature_names)
    
    logger.info(f"Discovered {len(sr_result.expressions)} expressions")
    if sr_result.best_expression:
        logger.info(f"Best expression: {sr_result.best_expression.expression}")
        logger.info(f"Best loss: {sr_result.best_expression.loss:.6f}")
        logger.info(f"Best complexity: {sr_result.best_expression.complexity}")
    
    # Step 6: Filter candidates
    logger.info("Step 6: Filtering candidates...")
    
    filter_config = FilterConfig(
        min_ic=0.01,  # Lower threshold for synthetic data
        min_ic_stability=0.5,
        min_sharpe=0.2,
        max_turnover=0.8,
        max_complexity=20
    )
    
    statistical_filter = StatisticalFilter(filter_config)
    
    # Use validation split (last 20% of data)
    split_idx = int(len(features_df) * 0.8)
    val_features = features_df.iloc[split_idx:]
    val_target = target_final.iloc[split_idx:]
    
    filter_results = statistical_filter.filter(
        sr_result.expressions,
        val_features,
        val_target
    )
    
    logger.info(f"{len(filter_results)} candidates passed the filter")
    
    # Step 7: Generate code
    logger.info("Step 7: Generating code...")
    
    for result in filter_results[:3]:  # Generate code for top 3 alphas
        alpha_expr = result.candidate
        code = generate_alpha_code(alpha_expr, feature_names)
        
        logger.info(f"\nGenerated code for alpha {alpha_expr.rank}:")
        logger.info("=" * 50)
        logger.info(code)
        logger.info("=" * 50)
    
    logger.info("Basic example completed!")
    
    return {
        'panel': panel,
        'features': all_features,
        'normalized_features': normalized_features,
        'sr_result': sr_result,
        'filter_results': filter_results
    }


if __name__ == "__main__":
    run_basic_example()
