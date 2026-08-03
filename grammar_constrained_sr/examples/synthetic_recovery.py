"""
Synthetic recovery test example.

This example demonstrates the synthetic recovery test described in Section 3.5
of the methodology. It validates that the grammar can express known alpha signals.
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
from src.validation.statistical import compute_ic


def run_synthetic_recovery_test():
    """Run the synthetic recovery test."""
    logger.info("Running synthetic recovery test...")
    
    # Test different planted alpha signals
    test_cases = [
        {
            'name': 'Mean Reversion',
            'planted_alpha': 'mean_reversion',
            'signal_strength': 0.15,
            'description': 'Simple mean-reversion signal: buy after negative returns'
        },
        {
            'name': 'Momentum',
            'planted_alpha': 'momentum',
            'signal_strength': 0.12,
            'description': 'Momentum signal: buy assets with positive past returns'
        },
        {
            'name': 'Volatility Anomaly',
            'planted_alpha': 'volatility',
            'signal_strength': 0.1,
            'description': 'Low volatility predicts higher returns'
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        logger.info(f"\nTesting {test_case['name']}: {test_case['description']}")
        
        # Generate synthetic data with planted alpha
        panel = generate_synthetic_panel(
            num_assets=50,
            num_days=1000,
            start_date="2020-01-01",
            planted_alpha=test_case['planted_alpha'],
            signal_strength=test_case['signal_strength'],
            seed=42
        )
        
        # Compute features
        feature_config = FeatureConfig(
            window_sizes=[5, 10, 20, 60],
            include_wavelet=False,  # Disable for speed
            include_hilbert=True,
            include_spectral=True
        )
        
        feature_factory = FeatureFactory(feature_config)
        features = feature_factory.compute_all_features(panel)
        
        # Normalize features
        norm_config = NormalizationConfig(
            include_raw=False,
            include_rank=True,
            include_zscore=True
        )
        
        normalizer = CombinedNormalizer(norm_config)
        normalized_features = normalizer.normalize(features)
        
        # Stack features
        features_stacked = normalized_features.stack()
        features_stacked.index = panel.index
        
        # Get target
        target = panel['forward_return']
        
        # Align
        common_index = features_stacked.index.intersection(target.index)
        features_final = features_stacked.loc[common_index]
        target_final = target.loc[common_index]
        
        # Convert to DataFrame
        features_df = features_final.unstack(level='symbol')
        
        # Run symbolic regression
        pysr_config = get_default_config()
        pysr_config.n_populations = 15
        pysr_config.population_size = 30
        pysr_config.max_iterations = 30
        pysr_config.max_time = 120
        
        regressor = SymbolicRegressor(pysr_config)
        feature_names = list(features_df.columns)
        
        sr_result = regressor.fit(features_df, target_final, feature_names)
        
        # Check recovery
        if sr_result.best_expression:
            # Compute correlation between discovered alpha and true signal
            # For this, we need to know what the true signal is
            # In our synthetic data, we planted a specific signal
            
            # For now, just check if we get reasonable IC
            best_alpha = sr_result.best_expression
            
            # Compute alpha values
            expr_str = best_alpha.expression
            for name in best_alpha.feature_names:
                expr_str = expr_str.replace(name, f"features_df['{name}']")
            
            try:
                alpha_values = eval(expr_str)
                ic = compute_ic(alpha_values, target_final)
                
                result = {
                    'test_case': test_case['name'],
                    'discovered_expression': best_alpha.expression,
                    'loss': best_alpha.loss,
                    'complexity': best_alpha.complexity,
                    'ic': ic,
                    'success': ic > 0.05  # Consider successful if IC > 0.05
                }
                
                results.append(result)
                
                logger.info(f"  Discovered: {best_alpha.expression}")
                logger.info(f"  Loss: {best_alpha.loss:.6f}")
                logger.info(f"  Complexity: {best_alpha.complexity}")
                logger.info(f"  IC: {ic:.4f}")
                logger.info(f"  Success: {'YES' if ic > 0.05 else 'NO'}")
                
            except Exception as e:
                logger.error(f"  Failed to evaluate expression: {str(e)}")
                results.append({
                    'test_case': test_case['name'],
                    'error': str(e),
                    'success': False
                })
        else:
            logger.warning(f"  No expressions discovered for {test_case['name']}")
            results.append({
                'test_case': test_case['name'],
                'error': 'No expressions discovered',
                'success': False
            })
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SYNTHETIC RECOVERY TEST SUMMARY")
    logger.info("=" * 60)
    
    successful = sum(1 for r in results if r.get('success', False))
    total = len(results)
    
    for result in results:
        status = "PASS" if result.get('success', False) else "FAIL"
        logger.info(f"{result['test_case']:20s} {status:6s} "
                   f"IC: {result.get('ic', 0):.4f}")
    
    logger.info(f"\nOverall: {successful}/{total} tests passed")
    
    return results


if __name__ == "__main__":
    run_synthetic_recovery_test()
