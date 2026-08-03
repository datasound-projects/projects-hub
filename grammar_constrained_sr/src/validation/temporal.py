"""
Temporal validation splits for alpha discovery.

This module implements strict temporal splitting for validation,
ensuring no lookahead bias and proper out-of-sample testing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class TemporalSplit:
    """Represents a temporal split of the data."""
    train_start: datetime
    train_end: datetime
    validation_start: datetime
    validation_end: datetime
    test_start: datetime
    test_end: datetime
    
    def __post_init__(self):
        # Ensure proper ordering
        assert self.train_end < self.validation_start, "Train end must be before validation start"
        assert self.validation_end < self.test_start, "Validation end must be before test start"


@dataclass
class TemporalSplitConfig:
    """Configuration for temporal splits."""
    # Proportions for each split
    train_ratio: float = 0.6
    validation_ratio: float = 0.2
    test_ratio: float = 0.2
    
    # Whether to use fixed dates or ratios
    use_fixed_dates: bool = False
    
    # Fixed dates (if use_fixed_dates is True)
    train_end_date: Optional[str] = None
    validation_end_date: Optional[str] = None
    
    # Minimum number of samples in each split
    min_samples: int = 100
    
    # Whether to shuffle (should be False for temporal splits)
    shuffle: bool = False


class TemporalSplitter:
    """
    Creates temporal splits for validation.
    
    Ensures strict temporal separation between train, validation, and test sets.
    """
    
    def __init__(self, 
                 config: Optional[TemporalSplitConfig] = None):
        """
        Initialize the TemporalSplitter.
        
        Args:
            config: Temporal split configuration
        """
        self.config = config or TemporalSplitConfig()
    
    def split(self, 
             data: pd.DataFrame,
             dates: Optional[pd.DatetimeIndex] = None) -> TemporalSplit:
        """
        Create temporal splits for the given data.
        
        Args:
            data: Input data (must have datetime index)
            dates: Optional datetime index (if not using data.index)
            
        Returns:
            TemporalSplit object
        """
        # Get dates from data or parameter
        if dates is None:
            if isinstance(data.index, pd.DatetimeIndex):
                dates = data.index
            elif isinstance(data.index, pd.MultiIndex):
                # Extract date level from MultiIndex
                date_level = [i for i, name in enumerate(data.index.names) if 'date' in name.lower()]
                if date_level:
                    dates = data.index.get_level_values(date_level[0])
                else:
                    dates = pd.DatetimeIndex(data.index.get_level_values(1))
            else:
                raise ValueError("Cannot extract dates from data index")
        
        # Sort dates
        dates = dates.sort_values()
        
        # Check if using fixed dates
        if self.config.use_fixed_dates:
            return self._split_fixed_dates(dates)
        else:
            return self._split_by_ratios(dates)
    
    def _split_by_ratios(self, 
                       dates: pd.DatetimeIndex) -> TemporalSplit:
        """Split by ratios."""
        n = len(dates)
        
        # Calculate split points
        train_end_idx = int(n * self.config.train_ratio)
        validation_end_idx = int(n * (self.config.train_ratio + self.config.validation_ratio))
        
        # Ensure minimum samples
        train_end_idx = max(train_end_idx, self.config.min_samples)
        validation_end_idx = max(validation_end_idx, train_end_idx + self.config.min_samples)
        validation_end_idx = min(validation_end_idx, n - self.config.min_samples)
        
        # Create split
        split = TemporalSplit(
            train_start=dates[0],
            train_end=dates[train_end_idx - 1],
            validation_start=dates[train_end_idx],
            validation_end=dates[validation_end_idx - 1],
            test_start=dates[validation_end_idx],
            test_end=dates[-1]
        )
        
        return split
    
    def _split_fixed_dates(self, 
                         dates: pd.DatetimeIndex) -> TemporalSplit:
        """Split by fixed dates."""
        if not self.config.train_end_date or not self.config.validation_end_date:
            raise ValueError("Fixed dates must be provided when use_fixed_dates is True")
        
        train_end = pd.to_datetime(self.config.train_end_date)
        validation_end = pd.to_datetime(self.config.validation_end_date)
        
        # Find indices
        train_end_idx = dates.get_loc(train_end, method='bfill')
        validation_end_idx = dates.get_loc(validation_end, method='bfill')
        
        split = TemporalSplit(
            train_start=dates[0],
            train_end=dates[train_end_idx],
            validation_start=dates[train_end_idx + 1],
            validation_end=dates[validation_end_idx],
            test_start=dates[validation_end_idx + 1],
            test_end=dates[-1]
        )
        
        return split
    
    def create_masks(self, 
                     data: pd.DataFrame,
                     split: TemporalSplit) -> Dict[str, pd.Series]:
        """
        Create boolean masks for each split.
        
        Args:
            data: Input data
            split: TemporalSplit object
            
        Returns:
            Dictionary with boolean masks for train, validation, test
        """
        # Get date index
        if isinstance(data.index, pd.DatetimeIndex):
            dates = data.index
        elif isinstance(data.index, pd.MultiIndex):
            date_level = [i for i, name in enumerate(data.index.names) if 'date' in name.lower()]
            if date_level:
                dates = data.index.get_level_values(date_level[0])
            else:
                dates = pd.DatetimeIndex(data.index.get_level_values(1))
        else:
            raise ValueError("Cannot extract dates from data index")
        
        # Create masks
        train_mask = (dates >= split.train_start) & (dates <= split.train_end)
        validation_mask = (dates > split.train_end) & (dates <= split.validation_end)
        test_mask = (dates > split.validation_end) & (dates <= split.test_end)
        
        return {
            'train': train_mask,
            'validation': validation_mask,
            'test': test_mask
        }
    
    def split_data(self, 
                   data: pd.DataFrame,
                   split: Optional[TemporalSplit] = None) -> Dict[str, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            data: Input data
            split: Optional TemporalSplit (will be created if not provided)
            
        Returns:
            Dictionary with train, validation, test DataFrames
        """
        if split is None:
            split = self.split(data)
        
        masks = self.create_masks(data, split)
        
        return {
            'train': data[masks['train']],
            'validation': data[masks['validation']],
            'test': data[masks['test']]
        }


def create_temporal_splits(
    data: pd.DataFrame,
    train_ratio: float = 0.6,
    validation_ratio: float = 0.2,
    test_ratio: float = 0.2
) -> TemporalSplit:
    """
    Convenience function to create temporal splits.
    
    Args:
        data: Input data
        train_ratio: Proportion for training
        validation_ratio: Proportion for validation
        test_ratio: Proportion for testing
        
    Returns:
        TemporalSplit object
    """
    config = TemporalSplitConfig(
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio
    )
    
    splitter = TemporalSplitter(config)
    return splitter.split(data)


def train_test_split_temporal(
    data: pd.DataFrame,
    test_ratio: float = 0.2,
    shuffle: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple train-test split with temporal ordering.
    
    Args:
        data: Input data
        test_ratio: Proportion for testing
        shuffle: Whether to shuffle (should be False for temporal)
        
    Returns:
        Tuple of (train, test) DataFrames
    """
    # Get dates
    if isinstance(data.index, pd.DatetimeIndex):
        dates = data.index
    elif isinstance(data.index, pd.MultiIndex):
        date_level = [i for i, name in enumerate(data.index.names) if 'date' in name.lower()]
        if date_level:
            dates = data.index.get_level_values(date_level[0])
        else:
            dates = pd.DatetimeIndex(data.index.get_level_values(1))
    else:
        raise ValueError("Cannot extract dates from data index")
    
    # Sort by date
    data = data.sort_index()
    
    # Calculate split point
    n = len(data)
    split_idx = int(n * (1 - test_ratio))
    
    # Split
    train = data.iloc[:split_idx]
    test = data.iloc[split_idx:]
    
    return train, test
