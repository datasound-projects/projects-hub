"""
Full pipeline example.

This example demonstrates the complete alpha discovery pipeline as described
in the methodology, including all stages from data loading to code generation.
"""

import pandas as pd
import numpy as np
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.pipeline.main import AlphaDiscoveryPipeline, PipelineConfig
from src.data.synthetic import generate_synthetic_panel


def run_full_pipeline_example():
    """Run the full pipeline example."""
    logger.info("Running full pipeline example...")
    
    # Generate synthetic data
    logger.info("Generating synthetic data...")
    panel = generate_synthetic_panel(
        num_assets=50,
        num_days=1000,
        start_date="2020-01-01",
        planted_alpha="mean_reversion",
        signal_strength=0.1,
        seed=42
    )
    
    logger.info(f"Generated panel with {len(panel.index.get_level_values('symbol').unique())} assets "
               f"and {len(panel.index.get_level_values('date').unique())} dates")
    
    # Create pipeline configuration
    pipeline_config = PipelineConfig(
        # Use smaller configuration for faster execution
        pysr_config__n_populations=10,
        pysr_config__population_size=20,
        pysr_config__max_iterations=30,
        pysr_config__max_time=120,
        
        # Feature configuration
        feature_config__window_sizes=[5, 10, 20, 60],
        feature_config__include_wavelet=False,  # Disable for speed
        feature_config__include_hilbert=True,
        feature_config__include_spectral=True,
        
        # Validation configuration
        validation_config__min_ic=0.01,
        validation_config__min_ic_stability=0.5,
        validation_config__min_sharpe=0.2,
        validation_config__max_turnover=0.8,
        
        # Run synthetic test
        run_synthetic_test=False,  # Skip for speed
        
        # Code generation
        codegen_config__language="python",
        codegen_config__include_type_hints=True,
        codegen_config__include_docstring=True
    )
    
    # Create and run pipeline
    start_time = time.time()
    
    pipeline = AlphaDiscoveryPipeline(pipeline_config)
    result = pipeline.run(data=panel)
    
    elapsed_time = time.time() - start_time
    
    logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE RESULTS SUMMARY")
    logger.info("=" * 60)
    
    logger.info(f"Number of discovered alphas: {len(result.alphas)}")
    logger.info(f"Number of filtered alphas: {len(result.filtered_alphas)}")
    
    if result.best_alpha:
        logger.info(f"Best alpha expression: {result.best_alpha.expression}")
        logger.info(f"Best alpha complexity: {result.best_alpha.complexity}")
    
    if result.filtered_alphas:
        logger.info("\nTop filtered alphas:")
        for i, filter_result in enumerate(result.filtered_alphas[:5]):
            alpha = filter_result.candidate
            logger.info(f"  {i+1}. {alpha.expression[:50]}... "
                       f"(IC: {filter_result.ic:.4f}, "
                       f"Sharpe: {filter_result.sharpe_ratio:.4f}, "
                       f"Complexity: {alpha.complexity})")
    
    if result.generated_code:
        logger.info("\nGenerated code for top alpha:")
        logger.info("=" * 60)
        # Get the first generated code
        first_code_key = list(result.generated_code.keys())[0]
        logger.info(result.generated_code[first_code_key])
        logger.info("=" * 60)
    
    # Print timing information
    logger.info("\nTiming breakdown:")
    for step, time_taken in result.timing.items():
        logger.info(f"  {step}: {time_taken:.2f}s")
    
    return result


if __name__ == "__main__":
    run_full_pipeline_example()
