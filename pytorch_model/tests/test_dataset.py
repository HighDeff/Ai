"""
Test suite for dataset loading and validation.

Tests cover:
- Vocabulary alignment
- Label range validation
- Train/test split correctness
- Batch output compatibility with model
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest
from dataset import Vocabulary, TextClassificationDataset, generate_sample_data, split_data
from datasets.topic_dataset import TopicDataset, TopicDataLoader, TopicSample
from datasets.imdb_dataset import IMDBDataset, IMDBDataLoader
from datasets.expected_results import (
    ExpectedResults, 
    IMDBExpectedResults, 
    TopicExpectedResults,
    create_synthetic_expected_results
)


class TestVocabulary:
    """Test vocabulary building and alignment."""

    def test_vocab_creation(self):
        """Test vocabulary can be created."""
        vocab = Vocabulary(min_freq=2, max_vocab_size=1000)
        assert vocab.vocab_size >= 5  # At least special tokens

    def test_vocab_build(self):
        """Test vocabulary building from texts."""
        texts = [
            "hello world",
            "machine learning",
            "deep neural networks",
            "natural language processing"
        ]
        
        vocab = Vocabulary(min_freq=1, max_vocab_size=100)
        vocab.build_vocab(texts)
        
        assert vocab.vocab_size > 5
        assert vocab.pad_idx == 0
        assert vocab.unk_idx == 1

    def test_vocab_encode_decode(self):
        """Test encoding and decoding."""
        vocab = Vocabulary(min_freq=1)
        vocab.build_vocab(["hello world", "test sentence"])
        
        text = "hello world"
        encoded = vocab.encode(text)
        decoded = vocab.decode(encoded)
        
        assert len(encoded) > 0
        assert "hello" in decoded or "world" in decoded or "?" in decoded

    def test_vocab_max_size(self):
        """Test vocabulary respects max size."""
        texts = ["word " + str(i) for i in range(100)]
        
        vocab = Vocabulary(min_freq=1, max_vocab_size=50)
        vocab.build_vocab(texts)
        
        assert vocab.vocab_size <= 50 + 5  # +5 for special tokens


class TestLabelRange:
    """Test label range validation."""

    def test_labels_in_range(self):
        """Verify labels are within valid range."""
        texts, labels = generate_sample_data(num_samples=100, num_classes=4)
        
        min_label = min(labels)
        max_label = max(labels)
        
        assert min_label >= 0
        assert max_label < 4

    def test_all_labels_present(self):
        """Verify all class labels are present."""
        texts, labels = generate_sample_data(num_samples=1000, num_classes=4)
        
        unique_labels = set(labels)
        
        # With enough samples, all classes should be present
        assert len(unique_labels) >= 2  # At least 2 classes represented

    def test_binary_labels(self):
        """Test binary classification labels."""
        texts, labels = generate_sample_data(num_samples=100, num_classes=2)
        
        for label in labels:
            assert label in [0, 1]


class TestTrainTestSplit:
    """Test train/test split functionality."""

    def test_split_ratio(self):
        """Verify split ratios sum to 1."""
        texts, labels = generate_sample_data(num_samples=100)
        
        train_t, train_l, val_t, val_l, test_t, test_l = split_data(
            texts, labels,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1
        )
        
        total = len(train_t) + len(val_t) + len(test_t)
        assert total == len(texts)

    def test_split_stratification(self):
        """Test that splits maintain some label distribution."""
        texts, labels = generate_sample_data(num_samples=100, num_classes=2)
        
        train_t, train_l, val_t, val_l, test_t, test_l = split_data(
            texts, labels, seed=42
        )
        
        # Both splits should have some labels
        assert len(train_l) > 0
        assert len(test_l) > 0

    def test_split_reproducibility(self):
        """Test that split is reproducible with same seed."""
        texts, labels = generate_sample_data(num_samples=50, seed=42)
        
        t1, l1, t2, l2, t3, l3 = split_data(texts, labels, seed=123)
        t1_, l1_, t2_, l2_, t3_, l3_ = split_data(texts, labels, seed=123)
        
        assert t1 == t1_
        assert l1 == l1_


class TestBatchOutput:
    """Test that batch output matches model requirements."""

    def test_batch_dimensions(self):
        """Test batch has correct dimensions."""
        vocab = Vocabulary(min_freq=1, max_vocab_size=1000)
        vocab.build_vocab(["test text " + str(i) for i in range(100)])
        
        texts, labels = generate_sample_data(num_samples=50, num_classes=3)
        
        dataset = TextClassificationDataset(
            texts=texts[:30],
            labels=labels[:30],
            vocab=vocab,
            max_length=32
        )
        
        input_ids, label = dataset[0]
        
        assert input_ids.dim() == 1
        assert label.dim() == 0
        assert label.item() in [0, 1, 2]

    def test_batch_padding(self):
        """Test that batches handle variable length correctly."""
        from dataset import DataCollator
        
        vocab = Vocabulary(min_freq=1, max_vocab_size=500)
        vocab.build_vocab(["sample text " + str(i) for i in range(20)])
        
        texts = ["short", "medium length text", "very long text here"]
        labels = [0, 1, 0]
        
        dataset = TextClassificationDataset(
            texts=texts,
            labels=labels,
            vocab=vocab,
            max_length=10
        )
        
        collator = DataCollator(vocab=vocab)
        batch = collator([dataset[i] for i in range(len(dataset))])
        
        assert batch["input_ids"].shape[0] == 3  # batch size
        assert batch["input_ids"].shape[1] == 10  # max length


class TestTopicDataset:
    """Test topic classification dataset."""

    def test_topic_sample_creation(self):
        """Test creating topic samples."""
        sample = TopicSample(
            question="What is AI?",
            context="Artificial Intelligence field",
            answer="Computer systems that mimic intelligence",
            label=0,
            topic_name="Technology"
        )
        
        assert sample.label == 0
        assert sample.topic_name == "Technology"

    def test_topic_data_loader(self):
        """Test topic data loader creates loaders."""
        loader = TopicDataLoader(vocab_size=1000, batch_size=8)
        train_loader, test_loader, vocab, samples = loader.load_topic_data()
        
        assert train_loader is not None
        assert test_loader is not None
        assert len(samples) > 0
        
        # Check a batch
        batch = next(iter(train_loader))
        assert "input_ids" in batch
        assert "labels" in batch

    def test_topic_expected_results(self):
        """Test expected results for topic dataset."""
        loader = TopicDataLoader(vocab_size=500)
        loader.load_topic_data()
        
        expected = loader.get_expected_results()
        
        assert expected.num_classes >= 2
        assert len(expected.class_names) >= 2


class TestIMDBDataset:
    """Test IMDB sentiment dataset."""

    def test_imdb_data_loader(self):
        """Test IMDB data loader creates loaders."""
        loader = IMDBDataLoader(vocab_size=1000, batch_size=8)
        train_loader, test_loader, vocab = loader.load_imdb()
        
        assert train_loader is not None
        assert test_loader is not None
        
        # Check a batch
        batch = next(iter(train_loader))
        assert "input_ids" in batch
        assert "labels" in batch

    def test_imdb_labels_binary(self):
        """Test IMDB labels are binary."""
        loader = IMDBDataLoader(vocab_size=500)
        train_loader, _, _ = loader.load_imdb()
        
        # Get first batch
        batch = next(iter(train_loader))
        labels = batch["labels"]
        
        for label in labels:
            assert label.item() in [0, 1]

    def test_imdb_expected_results(self):
        """Test expected results for IMDB."""
        from datasets.expected_results import IMDBExpectedResults
        
        expected = IMDBExpectedResults()
        
        assert expected.num_classes == 2
        assert "negative" in expected.class_names
        assert "positive" in expected.class_names


class TestExpectedResultsSchema:
    """Test expected results schema."""

    def test_create_expected_results(self):
        """Test creating expected results."""
        results = ExpectedResults(
            task_type="classification",
            dataset_name="Test",
            num_classes=3,
            class_names=["A", "B", "C"],
            sample_inputs=["input1", "input2"],
            expected_labels=[0, 1]
        )
        
        assert results.num_classes == 3
        assert len(results.sample_inputs) == 2

    def test_validation_predictions(self):
        """Test prediction validation."""
        results = ExpectedResults(
            task_type="classification",
            dataset_name="Test",
            num_classes=3,
            sample_inputs=["a", "b", "c"],
            expected_labels=[0, 1, 2]
        )
        
        predictions = [0, 1, 2]
        validation = results.validate_predictions(predictions)
        
        assert validation["valid"] == True
        assert validation["accuracy"] == 1.0

    def test_validation_partial_correct(self):
        """Test validation with partial correctness."""
        results = ExpectedResults(
            task_type="classification",
            dataset_name="Test",
            num_classes=3,
            sample_inputs=["a", "b", "c", "d"],
            expected_labels=[0, 1, 2, 0]
        )
        
        predictions = [0, 1, 0, 1]  # 2 correct out of 4
        validation = results.validate_predictions(predictions)
        
        assert validation["valid"] == True
        assert validation["accuracy"] == 0.5

    def test_save_load(self, tmp_path):
        """Test saving and loading expected results."""
        results = ExpectedResults(
            task_type="classification",
            dataset_name="Test",
            num_classes=2,
            sample_inputs=["test"],
            expected_labels=[0]
        )
        
        path = tmp_path / "test_expected.json"
        results.save(str(path))
        
        loaded = ExpectedResults.load(str(path))
        
        assert loaded.dataset_name == "Test"
        assert loaded.num_classes == 2


class TestSyntheticData:
    """Test synthetic data generation."""

    def test_generate_sample_data(self):
        """Test synthetic data generation."""
        texts, labels = generate_sample_data(num_samples=100, num_classes=4)
        
        assert len(texts) == 100
        assert len(labels) == 100
        assert len(texts) == len(labels)

    def test_synthetic_expected_results(self):
        """Test synthetic expected results creation."""
        results = create_synthetic_expected_results(num_samples=50, num_classes=3)
        
        assert results.dataset_name == "Synthetic"
        assert len(results.sample_inputs) == 50
        assert len(results.expected_labels) == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
