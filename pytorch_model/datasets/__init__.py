"""
Dataset Loaders and Expected Results for PyTorch Model.

Includes:
- IMDB sentiment classification dataset
- Topic classification dataset with Q&A format
- Expected results schema for validation

Example:
    >>> from datasets import IMDBDataLoader
    >>> loader = IMDBDataLoader()
    >>> train_loader, test_loader, vocab = loader.load_imdb()
"""

from .imdb_dataset import IMDBDataset, IMDBDataLoader, IMDBExpectedResults
from .topic_dataset import TopicDataset, TopicDataLoader, TopicSample, TopicExpectedResults
from .expected_results import (
    ExpectedResults,
    TaskType,
    DatasetType,
    ExpectedResultsRegistry,
    create_synthetic_expected_results,
    validate_dataset_format,
)

__all__ = [
    # IMDB
    "IMDBDataset",
    "IMDBDataLoader", 
    "IMDBExpectedResults",
    # Topic
    "TopicDataset",
    "TopicDataLoader",
    "TopicSample",
    "TopicExpectedResults",
    # Expected Results
    "ExpectedResults",
    "TaskType",
    "DatasetType",
    "ExpectedResultsRegistry",
    "create_synthetic_expected_results",
    "validate_dataset_format",
]
