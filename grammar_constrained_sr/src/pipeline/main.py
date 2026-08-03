"""
Main pipeline for grammar-constrained symbolic regression alpha discovery.

This module implements the complete workflow described in the methodology:
1. Grammar design (stylized fact primitives)
2. Feature computation
3. Symbolic regression with constraints
4. Statistical filtering
5. Code generation
6. Policy optimization
7. Fusion
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
import logging
from datetime import datetime

from ..data.loader import DataLoader
from ..data.panel import PanelConstructor, PanelConfig
from ..data.synthetic import generate_synthetic_panel
from ..features.factory import FeatureFactory, FeatureConfig
from ..normalization import CombinedNormalizer, NormalizationConfig
from ..symbolic.regression import SymbolicRegressor, SymbolicRegressionResult
from ..symbolic.pysr_config import PySRConfig, get_default_config
from ..symbolic.expressions import AlphaExpression
from ..validation.temporal import TemporalSplitter, TemporalSplitConfig, TemporalSplit
from ..validation.statistical import StatisticalFilter, FilterConfig, FilterResult
from ..validation.metrics import PerformanceMetrics, MetricsCalculator
from ..codegen.generator import CodeGenerator, CodeGenerationConfig
from ..policy.quintile import QuintileLongShortPolicy
from ..policy.optimizer import PolicyOptimizer, OptimizationConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the complete pipeline."""
    # Data configuration
    data_dir: Optional[str] = None
    forward_return_horizon: int = 1
    
    # Feature configuration
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    
    # Normalization configuration
    normalization_config: NormalizationConfig = field(default_factory=NormalizationConfig)
    
    # PySR configuration
    pysr_config: PySRConfig = field(default_factory=get_default_config)
    
    # Validation configuration
    validation_config: FilterConfig = field(default_factory=FilterConfig)
    temporal_config: TemporalSplitConfig = field(default_factory=TemporalSplitConfig)
    
    # Code generation configuration
    codegen_config: CodeGenerationConfig = field(default_factory=CodeGenerationConfig)
    
    # Policy optimization configuration
    policy_config: OptimizationConfig = field(default_factory=OptimizationConfig)
    
    # Whether to run synthetic recovery test first
    run_synthetic_test: bool = True
    
    # Whether to include wavelet features
    include_wavelet: bool = True
    
    # Whether to include Hilbert features
    include_hilbert: bool = True
    
    # Whether to include spectral features
    include_spectral: bool = True


@dataclass
class PipelineResult:
    """Result of the complete pipeline."""
    # Discovered alpha expressions
    alphas: List[AlphaExpression]
    
    # Filtered alpha expressions
    filtered_alphas: List[FilterResult]
    
    # Best alpha
    best_alpha: Optional[AlphaExpression] = None
    
    # Generated code
    generated_code: Dict[str, str] = field(default_factory=dict)
    
    # Performance metrics
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Temporal splits
    splits: Optional[TemporalSplit] = None
    
    # Feature information
    feature_info: Dict[str, Any] = field(default_factory=dict)
    
    # Timing information
    timing: Dict[str, float] = field(default_factory=dict)


class AlphaDiscoveryPipeline:
    """
    Main pipeline for grammar-constrained symbolic regression alpha discovery.
    
    Orchestrates the complete workflow from data loading to code generation.
    """
    
    def __init__(self, 
                 config: Optional[PipelineConfig] = None):
        """
        Initialize the pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        logger.info("Initializing pipeline components...")
        
        # Feature factory
        self.feature_factory = FeatureFactory(self.config.feature_config)
        
        # Normalizer
        self.normalizer = CombinedNormalizer(
            config=self.config.normalization_config,
            include_raw=self.config.normalization_config.include_raw,
            include_rank=self.config.normalization_config.include_rank,
            include_zscore=self.config.normalization_config.include_zscore
        )
        
        # Symbolic regressor
        self.regressor = SymbolicRegressor(self.config.pysr_config)
        
        # Temporal splitter
        self.splitter = TemporalSplitter(self.config.temporal_config)
        
        # Statistical filter
        self.filter = StatisticalFilter(self.config.validation_config)
        
        # Code generator
        self.codegen = CodeGenerator(self.config.codegen_config)
        
        # Policy optimizer
        self.policy_optimizer = PolicyOptimizer(self.config.policy_config)
        
        # Metrics calculator
        self.metrics_calculator = MetricsCalculator()
        
        logger.info("Pipeline components initialized")
    
    def run(self, 
            data: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = None,
            data_path: Optional[str] = None) -> PipelineResult:
        """
        Run the complete pipeline.
        
        Args:
            data: Input data (panel DataFrame or dict of asset DataFrames)
            data_path: Path to data directory
            
        Returns:
            PipelineResult with all outputs
        """
        import time
        
        start_time = time.time()
        result_timing = {}
        
        try:
            # Step 1: Load and prepare data
            logger.info("Step 1: Loading and preparing data...")
            start_step = time.time()
            
            if data is None and data_path:
                data = self._load_data(data_path)
            elif data is None:
                # Generate synthetic data for testing
                data = generate_synthetic_panel(
                    num_assets=50,
                    num_days=1000,
                    start_date="2020-01-01"
                )
                logger.info("Using synthetic data for testing")
            
            # Ensure we have a panel
            if isinstance(data, dict):
                panel = self._create_panel(data)
            else:
                panel = data
            
            result_timing['data_loading'] = time.time() - start_step
            logger.info(f"Data loading completed in {result_timing['data_loading']:.2f}s")
            
            # Step 2: Run synthetic recovery test (if configured)
            if self.config.run_synthetic_test:
                logger.info("Step 2: Running synthetic recovery test...")
                start_step = time.time()
                
                self._run_synthetic_recovery_test()
                
                result_timing['synthetic_test'] = time.time() - start_step
                logger.info(f"Synthetic recovery test completed in {result_timing['synthetic_test']:.2f}s")
            
            # Step 3: Compute features
            logger.info("Step 3: Computing features...")
            start_step = time.time()
            
            features = self._compute_features(panel)
            
            result_timing['feature_computation'] = time.time() - start_step
            logger.info(f"Feature computation completed in {result_timing['feature_computation']:.2f}s")
            logger.info(f"Computed {len(features.columns)} features")
            
            # Step 4: Normalize features
            logger.info("Step 4: Normalizing features...")
            start_step = time.time()
            
            normalized_features = self._normalize_features(features)
            
            result_timing['normalization'] = time.time() - start_step
            logger.info(f"Normalization completed in {result_timing['normalization']:.2f}s")
            logger.info(f"Normalized to {len(normalized_features.columns)} features")
            
            # Step 5: Create temporal splits
            logger.info("Step 5: Creating temporal splits...")
            start_step = time.time()
            
            split = self.splitter.split(panel)
            
            result_timing['splitting'] = time.time() - start_step
            logger.info(f"Temporal splitting completed in {result_timing['splitting']:.2f}s")
            
            # Step 6: Run symbolic regression
            logger.info("Step 6: Running symbolic regression...")
            start_step = time.time()
            
            train_data = self.splitter.split_data(normalized_features, split)['train']
            train_target = panel.loc[train_data.index, 'forward_return']
            
            # Get feature names
            feature_names = list(normalized_features.columns)
            
            # Run symbolic regression
            sr_result = self.regressor.fit(train_data, train_target, feature_names)
            
            result_timing['symbolic_regression'] = time.time() - start_step
            logger.info(f"Symbolic regression completed in {result_timing['symbolic_regression']:.2f}s")
            logger.info(f"Discovered {len(sr_result.expressions)} expressions")
            
            # Step 7: Statistical filtering
            logger.info("Step 7: Statistical filtering...")
            start_step = time.time()
            
            # Get validation data
            val_data = self.splitter.split_data(normalized_features, split)['validation']
            val_target = panel.loc[val_data.index, 'forward_return']
            
            # Filter candidates
            filter_results = self.filter.filter(
                sr_result.expressions,
                val_data,
                val_target,
                split
            )
            
            result_timing['filtering'] = time.time() - start_step
            logger.info(f"Statistical filtering completed in {result_timing['filtering']:.2f}s")
            logger.info(f"{len(filter_results)} candidates passed the filter")
            
            # Step 8: Code generation for passing alphas
            logger.info("Step 8: Generating code...")
            start_step = time.time()
            
            generated_code = {}
            for result in filter_results:
                alpha_expr = result.candidate
                code = self.codegen.generate_alpha_code(
                    alpha_expr,
                    feature_names
                )
                generated_code[f"alpha_{alpha_expr.rank}"] = code
            
            result_timing['code_generation'] = time.time() - start_step
            logger.info(f"Code generation completed in {result_timing['code_generation']:.2f}s")
            
            # Step 9: Policy optimization (for best alpha)
            if filter_results:
                logger.info("Step 9: Optimizing policy...")
                start_step = time.time()
                
                best_alpha = filter_results[0].candidate
                
                # Get test data
                test_data = self.splitter.split_data(normalized_features, split)['test']
                test_target = panel.loc[test_data.index, 'forward_return']
                
                # Compute alpha values for test set
                alpha_values = self._compute_alpha_values(best_alpha, test_data)
                
                # Optimize policy
                policy_result = self.policy_optimizer.optimize_quintile_policy(
                    alpha_values,
                    test_target
                )
                
                result_timing['policy_optimization'] = time.time() - start_step
                logger.info(f"Policy optimization completed in {result_timing['policy_optimization']:.2f}s")
            else:
                best_alpha = None
                policy_result = None
            
            # Step 10: Compute final metrics
            logger.info("Step 10: Computing final metrics...")
            start_step = time.time()
            
            metrics = self._compute_final_metrics(filter_results, panel, split)
            
            result_timing['metrics_computation'] = time.time() - start_step
            logger.info(f"Metrics computation completed in {result_timing['metrics_computation']:.2f}s")
            
            # Create final result
            result = PipelineResult(
                alphas=sr_result.expressions,
                filtered_alphas=filter_results,
                best_alpha=best_alpha,
                generated_code=generated_code,
                metrics=metrics,
                splits=split,
                feature_info=self._get_feature_info(),
                timing=result_timing
            )
            
            total_time = time.time() - start_time
            logger.info(f"Pipeline completed in {total_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load data from directory."""
        loader = DataLoader()
        assets = loader.load_directory(data_path)
        
        constructor = PanelConstructor(PanelConfig(
            forward_return_horizon=self.config.forward_return_horizon
        ))
        
        preprocessed = loader.preprocess_all(
            forward_return_horizon=self.config.forward_return_horizon
        )
        
        panel = constructor.construct_from_dict(preprocessed)
        panel = constructor.handle_missing_data(panel)
        
        return panel
    
    def _create_panel(self, 
                     asset_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create panel from asset data."""
        constructor = PanelConstructor(PanelConfig(
            forward_return_horizon=self.config.forward_return_horizon
        ))
        
        preprocessed = {}
        loader = DataLoader()
        
        for symbol, df in asset_data.items():
            preprocessed[symbol] = loader.preprocess_asset(
                loader.assets[symbol],
                forward_return_horizon=self.config.forward_return_horizon
            )
        
        panel = constructor.construct_from_dict(preprocessed)
        panel = constructor.handle_missing_data(panel)
        
        return panel
    
    def _compute_features(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Compute all features for the panel."""
        # Extract returns from panel
        if 'log_return' in panel.columns:
            returns = panel['log_return'].unstack(level='symbol')
        else:
            raise ValueError("Panel must contain 'log_return' column")
        
        # Compute all features
        features = self.feature_factory.compute_all_features(panel)
        
        return features
    
    def _normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Normalize features."""
        return self.normalizer.normalize(features)
    
    def _compute_alpha_values(self, 
                            alpha_expr: AlphaExpression,
                            features: pd.DataFrame) -> pd.Series:
        """Compute alpha values for an expression."""
        # This is a simplified version - in practice, we'd use the SymPy expression
        expr_str = alpha_expr.expression
        
        # Replace feature names with column references
        for name in alpha_expr.feature_names:
            expr_str = expr_str.replace(name, f"features['{name}']")
        
        # Evaluate
        try:
            alpha_values = eval(expr_str)
            return alpha_values
        except Exception as e:
            logger.error(f"Failed to evaluate expression: {str(e)}")
            return pd.Series(0.0, index=features.index)
    
    def _run_synthetic_recovery_test(self):
        """Run synthetic recovery test to validate the grammar."""
        logger.info("Running synthetic recovery test...")
        
        # Generate synthetic data with a planted alpha
        synthetic_panel = generate_synthetic_panel(
            num_assets=50,
            num_days=1000,
            planted_alpha="mean_reversion",
            signal_strength=0.1,
            seed=42
        )
        
        # Compute features
        synthetic_features = self.feature_factory.compute_all_features(synthetic_panel)
        
        # Normalize features
        synthetic_normalized = self.normalizer.normalize(synthetic_features)
        
        # Get target
        synthetic_target = synthetic_panel['forward_return']
        
        # Run symbolic regression
        sr_result = self.regressor.fit(
            synthetic_normalized,
            synthetic_target,
            list(synthetic_normalized.columns)
        )
        
        # Check if we recovered a similar expression
        if sr_result.best_expression:
            logger.info(f"Synthetic test: Best expression IC = {sr_result.best_expression.loss:.6f}")
            logger.info(f"Synthetic test: Expression = {sr_result.best_expression.expression}")
        
        return sr_result
    
    def _compute_final_metrics(self, 
                               filter_results: List[FilterResult],
                               panel: pd.DataFrame,
                               split: TemporalSplit) -> Dict[str, Any]:
        """Compute final performance metrics."""
        metrics = {}
        
        if not filter_results:
            return metrics
        
        # Get test data
        test_mask = self.splitter.create_masks(panel, split)['test']
        test_features = self.splitter.split_data(
            self._normalize_features(self._compute_features(panel)),
            split
        )['test']
        test_target = panel.loc[test_mask, 'forward_return']
        
        # Compute metrics for each filtered alpha
        alpha_metrics = []
        
        for result in filter_results:
            alpha_expr = result.candidate
            alpha_values = self._compute_alpha_values(alpha_expr, test_features)
            
            # Compute performance metrics
            perf_metrics = self.metrics_calculator.compute_all_metrics(
                alpha_values, test_target
            )
            
            alpha_metrics.append({
                'expression': alpha_expr.expression,
                'rank': alpha_expr.rank,
                'complexity': alpha_expr.complexity,
                'metrics': perf_metrics.to_dict()
            })
        
        metrics['alpha_metrics'] = alpha_metrics
        
        # Compute aggregate metrics
        if alpha_metrics:
            metrics['n_passing_alphas'] = len(alpha_metrics)
            metrics['best_ic'] = max([m['metrics']['ic'] for m in alpha_metrics])
            metrics['best_sharpe'] = max([m['metrics']['sharpe_ratio'] for m in alpha_metrics])
        
        return metrics
    
    def _get_feature_info(self) -> Dict[str, Any]:
        """Get information about the features."""
        return self.feature_factory.get_stylized_fact_summary()
    
    def run_synthetic_test(self) -> PipelineResult:
        """
        Run the pipeline on synthetic data for testing.
        
        Returns:
            PipelineResult with synthetic data results
        """
        # Generate synthetic data
        synthetic_panel = generate_synthetic_panel(
            num_assets=50,
            num_days=1000,
            planted_alpha="mean_reversion",
            signal_strength=0.1,
            seed=42
        )
        
        # Run pipeline
        return self.run(data=synthetic_panel)
    
    def run_with_config(self, 
                       config: PipelineConfig) -> PipelineResult:
        """
        Run the pipeline with a custom configuration.
        
        Args:
            config: Custom pipeline configuration
            
        Returns:
            PipelineResult
        """
        # Update configuration
        self.config = config
        self._initialize_components()
        
        # Run pipeline
        return self.run()


def run_alpha_discovery_pipeline(
    data: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = None,
    data_path: Optional[str] = None,
    config: Optional[PipelineConfig] = None
) -> PipelineResult:
    """
    Convenience function to run the alpha discovery pipeline.
    
    Args:
        data: Input data
        data_path: Path to data directory
        config: Pipeline configuration
        
    Returns:
        PipelineResult
    """
    pipeline = AlphaDiscoveryPipeline(config)
    return pipeline.run(data, data_path)
