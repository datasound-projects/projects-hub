"""
Workflow manager for alpha discovery pipeline.

This module provides a higher-level interface for managing the complete
alpha discovery workflow, including multiple runs, parameter sweeps,
and result aggregation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
import logging
from datetime import datetime
import time

from .main import AlphaDiscoveryPipeline, PipelineConfig, PipelineResult

logger = logging.getLogger(__name__)


@dataclass
class WorkflowConfig:
    """Configuration for workflow management."""
    # Base pipeline configuration
    base_config: PipelineConfig = field(default_factory=PipelineConfig)
    
    # Number of runs to perform
    n_runs: int = 1
    
    # Whether to run with different random seeds
    vary_seeds: bool = False
    
    # Whether to run with different temporal splits
    vary_splits: bool = False
    
    # Whether to run with different feature sets
    vary_features: bool = False
    
    # Output directory
    output_dir: Optional[str] = None
    
    # Whether to save intermediate results
    save_intermediate: bool = False


@dataclass
class WorkflowResult:
    """Result of a complete workflow."""
    # List of pipeline results
    results: List[PipelineResult]
    
    # Aggregated metrics
    aggregated_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Best alpha across all runs
    best_alpha: Optional[Any] = None
    
    # Timing information
    timing: Dict[str, float] = field(default_factory=dict)
    
    # Configuration used
    config: Optional[WorkflowConfig] = None


class WorkflowManager:
    """
    Manager for running multiple pipeline executions.
    
    Provides functionality for:
    - Running multiple pipeline executions
    - Varying parameters across runs
    - Aggregating results
    - Saving and loading results
    """
    
    def __init__(self, 
                 config: Optional[WorkflowConfig] = None):
        """
        Initialize the WorkflowManager.
        
        Args:
            config: Workflow configuration
        """
        self.config = config or WorkflowConfig()
        self.results: List[PipelineResult] = []
    
    def run(self, 
            data: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = None,
            data_path: Optional[str] = None) -> WorkflowResult:
        """
        Run the workflow.
        
        Args:
            data: Input data
            data_path: Path to data directory
            
        Returns:
            WorkflowResult
        """
        start_time = time.time()
        workflow_timing = {}
        
        try:
            # Run multiple pipeline executions
            for i in range(self.config.n_runs):
                logger.info(f"Running pipeline execution {i + 1}/{self.config.n_runs}")
                
                start_run = time.time()
                
                # Create pipeline with appropriate configuration
                if self.config.vary_seeds and i > 0:
                    # Vary the random seed
                    pipeline_config = self._vary_config_seed(self.config.base_config, i)
                elif self.config.vary_splits and i > 0:
                    # Vary the temporal splits
                    pipeline_config = self._vary_config_splits(self.config.base_config, i)
                elif self.config.vary_features and i > 0:
                    # Vary the feature set
                    pipeline_config = self._vary_config_features(self.config.base_config, i)
                else:
                    pipeline_config = self.config.base_config
                
                # Create and run pipeline
                pipeline = AlphaDiscoveryPipeline(pipeline_config)
                result = pipeline.run(data, data_path)
                
                # Save intermediate results if configured
                if self.config.save_intermediate and self.config.output_dir:
                    self._save_result(result, i)
                
                self.results.append(result)
                
                workflow_timing[f'run_{i}'] = time.time() - start_run
                logger.info(f"Run {i + 1} completed in {workflow_timing[f'run_{i}']:.2f}s")
            
            # Aggregate results
            aggregated_metrics = self._aggregate_results()
            
            # Find best alpha
            best_alpha = self._find_best_alpha()
            
            workflow_timing['total'] = time.time() - start_time
            
            # Create workflow result
            workflow_result = WorkflowResult(
                results=self.results,
                aggregated_metrics=aggregated_metrics,
                best_alpha=best_alpha,
                timing=workflow_timing,
                config=self.config
            )
            
            logger.info(f"Workflow completed in {workflow_timing['total']:.2f}s")
            logger.info(f"Best alpha: {best_alpha}")
            
            return workflow_result
            
        except Exception as e:
            logger.error(f"Workflow failed: {str(e)}")
            raise
    
    def _vary_config_seed(self, 
                         base_config: PipelineConfig,
                         run_index: int) -> PipelineConfig:
        """Vary the random seed for a run."""
        # Create a copy of the base config
        import copy
        config = copy.deepcopy(base_config)
        
        # Vary the PySR random seed
        if config.pysr_config.random_seed is not None:
            config.pysr_config.random_seed = 42 + run_index
        
        return config
    
    def _vary_config_splits(self, 
                           base_config: PipelineConfig,
                           run_index: int) -> PipelineConfig:
        """Vary the temporal splits for a run."""
        import copy
        config = copy.deepcopy(base_config)
        
        # Vary the train/validation/test ratios
        ratios = [
            (0.5, 0.25, 0.25),
            (0.6, 0.2, 0.2),
            (0.7, 0.15, 0.15),
            (0.55, 0.2, 0.25),
            (0.65, 0.2, 0.15)
        ]
        
        ratio = ratios[run_index % len(ratios)]
        config.temporal_config.train_ratio = ratio[0]
        config.temporal_config.validation_ratio = ratio[1]
        config.temporal_config.test_ratio = ratio[2]
        
        return config
    
    def _vary_config_features(self, 
                              base_config: PipelineConfig,
                              run_index: int) -> PipelineConfig:
        """Vary the feature set for a run."""
        import copy
        config = copy.deepcopy(base_config)
        
        # Toggle feature inclusion
        feature_config = config.feature_config
        
        if run_index % 2 == 0:
            feature_config.include_wavelet = True
        else:
            feature_config.include_wavelet = False
        
        if run_index % 3 == 0:
            feature_config.include_hilbert = True
        else:
            feature_config.include_hilbert = False
        
        if run_index % 4 == 0:
            feature_config.include_spectral = True
        else:
            feature_config.include_spectral = False
        
        return config
    
    def _aggregate_results(self) -> Dict[str, Any]:
        """Aggregate results from multiple runs."""
        if not self.results:
            return {}
        
        aggregated = {
            'n_runs': len(self.results),
            'total_alphas': sum(len(r.alphas) for r in self.results),
            'total_filtered_alphas': sum(len(r.filtered_alphas) for r in self.results),
            'avg_filtered_per_run': np.mean([len(r.filtered_alphas) for r in self.results]),
            'best_ic_across_runs': max(
                [max([f.candidate.loss for f in r.filtered_alphas], default=0) 
                 for r in self.results]
            ),
            'run_timings': [r.timing.get('total', 0) for r in self.results]
        }
        
        return aggregated
    
    def _find_best_alpha(self) -> Optional[Any]:
        """Find the best alpha across all runs."""
        if not self.results:
            return None
        
        best_alpha = None
        best_ic = -np.inf
        
        for result in self.results:
            for filter_result in result.filtered_alphas:
                if filter_result.ic > best_ic:
                    best_ic = filter_result.ic
                    best_alpha = filter_result.candidate
        
        return best_alpha
    
    def _save_result(self, 
                    result: PipelineResult,
                    run_index: int) -> None:
        """Save a pipeline result."""
        if not self.config.output_dir:
            return
        
        import os
        import json
        import pickle
        
        # Create output directory if it doesn't exist
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Save as JSON (summary)
        summary = {
            'run_index': run_index,
            'n_alphas': len(result.alphas),
            'n_filtered': len(result.filtered_alphas),
            'best_alpha': result.best_alpha.expression if result.best_alpha else None,
            'timing': result.timing
        }
        
        with open(os.path.join(self.config.output_dir, f'result_{run_index}_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save full result as pickle
        with open(os.path.join(self.config.output_dir, f'result_{run_index}.pkl'), 'wb') as f:
            pickle.dump(result, f)
        
        logger.info(f"Saved result {run_index} to {self.config.output_dir}")
    
    def load_results(self, 
                    directory: str) -> List[PipelineResult]:
        """
        Load saved results from a directory.
        
        Args:
            directory: Directory containing saved results
            
        Returns:
            List of PipelineResult objects
        """
        import os
        import pickle
        
        results = []
        
        for filename in os.listdir(directory):
            if filename.endswith('.pkl'):
                filepath = os.path.join(directory, filename)
                try:
                    with open(filepath, 'rb') as f:
                        result = pickle.load(f)
                        results.append(result)
                except Exception as e:
                    logger.error(f"Failed to load {filename}: {str(e)}")
        
        return results
    
    def compare_configurations(self, 
                              configs: List[PipelineConfig],
                              data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                              data_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Compare multiple configurations.
        
        Args:
            configs: List of pipeline configurations to compare
            data: Input data
            data_path: Path to data directory
            
        Returns:
            Dictionary with comparison results
        """
        comparison = {}
        
        for i, config in enumerate(configs):
            logger.info(f"Running configuration {i + 1}/{len(configs)}")
            
            # Create workflow config
            workflow_config = WorkflowConfig(
                base_config=config,
                n_runs=1,
                output_dir=None
            )
            
            # Create workflow manager
            manager = WorkflowManager(workflow_config)
            
            # Run workflow
            result = manager.run(data, data_path)
            
            # Store results
            comparison[f'config_{i}'] = {
                'config': config,
                'result': result,
                'n_filtered_alphas': len(result.results[0].filtered_alphas) if result.results else 0,
                'best_ic': max([f.candidate.loss for f in result.results[0].filtered_alphas], default=0) if result.results and result.results[0].filtered_alphas else 0
            }
        
        return comparison


def run_workflow(data: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = None,
                data_path: Optional[str] = None,
                config: Optional[WorkflowConfig] = None) -> WorkflowResult:
    """
    Convenience function to run a workflow.
    
    Args:
        data: Input data
        data_path: Path to data directory
        config: Workflow configuration
        
    Returns:
        WorkflowResult
    """
    manager = WorkflowManager(config)
    return manager.run(data, data_path)
