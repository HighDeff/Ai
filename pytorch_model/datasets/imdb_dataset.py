"""
IMDB Sentiment Classification Dataset Loader.

Supports:
- Loading IMDB dataset from CSV or HuggingFace datasets
- Binary sentiment classification (positive/negative)
- Train/test splitting
- Integration with existing Dataset class

Expected format:
- Text: Movie review string
- Label: 0 (negative) or 1 (positive)
"""

import os
import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset import Vocabulary, DataCollator


@dataclass
class IMDBSample:
    """Single IMDB sample."""
    text: str
    label: int  # 0 = negative, 1 = positive
    review_id: Optional[int] = None


@dataclass
class IMDBExpectedResults:
    """Expected results for IMDB dataset."""
    task_type: str = "classification"
    dataset_name: str = "IMDB"
    num_classes: int = 2
    class_names: List[str] = None
    
    def __post_init__(self):
        if self.class_names is None:
            self.class_names = ["negative", "positive"]


class IMDBDataset(Dataset):
    """
    IMDB Sentiment Dataset for binary text classification.
    
    Loads data from CSV file or generates sample data for testing.
    
    Args:
        texts: List of review texts
        labels: List of labels (0 or 1)
        vocab: Vocabulary for tokenization
        max_length: Maximum sequence length
        transform: Optional text transform
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        vocab: Vocabulary,
        max_length: int = 256,
        transform: Optional[callable] = None,
    ):
        assert len(texts) == len(labels), "Texts and labels must have same length"
        
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample."""
        text = self.texts[idx]
        label = self.labels[idx]
        
        if self.transform:
            text = self.transform(text)
        
        token_ids = self.vocab.encode(text, max_length=self.max_length)
        
        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )


class IMDBDataLoader:
    """
    Data loader for IMDB dataset with built-in train/test split.
    
    Example:
        >>> loader = IMDBDataLoader(vocab_size=10000)
        >>> train_loader, test_loader = loader.load_imdb()
        >>> for batch in train_loader:
        ...     print(batch['input_ids'].shape)
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        max_length: int = 256,
        batch_size: int = 32,
        train_ratio: float = 0.8,
        seed: int = 42,
    ):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.seed = seed
        
        self.vocab = None
        self.train_loader = None
        self.test_loader = None
    
    def _generate_sample_data(self) -> Tuple[List[str], List[int]]:
        """Generate sample IMDB-like data for testing."""
        random.seed(self.seed)
        
        positive_reviews = [
            "This movie was absolutely fantastic! Great acting and storyline.",
            "I loved this film. The director did an amazing job with the visuals.",
            "Best movie I've seen this year. Highly recommend to everyone!",
            "A masterpiece of cinema. The performances were outstanding.",
            "Wonderful and inspiring. This film touched my heart deeply.",
            "Excellent movie with great cinematography and sound design.",
            "A brilliant story beautifully told. Will watch again!",
            "Truly exceptional film-making. Every scene was perfect.",
            "This movie exceeded all my expectations. Five stars!",
            "Fantastic acting by all cast members. Plot was engaging.",
            "One of the best films in the genre. Very entertaining.",
            "Amazing storyline with unexpected twists. Loved every minute.",
            "Great film for all ages. The whole family enjoyed it.",
            "Professional and creative. This movie deserves recognition.",
            "Emotionally powerful and beautifully crafted masterpiece.",
        ]
        
        negative_reviews = [
            "This movie was terrible. Waste of time and money.",
            "I hated this film. The plot made no sense at all.",
            "One of the worst movies I've ever seen. Very disappointing.",
            "Awful acting and poor direction. Do not recommend.",
            "Boring and predictable. I fell asleep halfway through.",
            "This film was a disaster. Nothing went right here.",
            "Terrible screenplay and boring scenes throughout.",
            "I regret watching this movie. It was completely pointless.",
            "Poor production quality and bad editing made it unwatchable.",
            "The worst film of the year. Save your money!",
            "Extremely disappointing. The trailer was better than the movie.",
            "No storyline, bad acting, and terrible sound. Skip it!",
            "This movie had no redeeming qualities whatsoever.",
            "Completely flat and uninteresting. I left early.",
            "Horrible movie experience. Very poorly made film.",
        ]
        
        texts = []
        labels = []
        
        # Generate more samples by repeating with variations
        for _ in range(50):
            for review in positive_reviews:
                # Add some variation
                words = review.lower().split()
                random.shuffle(words)
                texts.append(" ".join(words[:8]) + " .")
                labels.append(1)
            
            for review in negative_reviews:
                words = review.lower().split()
                random.shuffle(words)
                texts.append(" ".join(words[:8]) + " .")
                labels.append(0)
        
        return texts, labels
    
    def _try_load_from_huggingface(self) -> Tuple[List[str], List[int]]:
        """Try to load IMDB from HuggingFace datasets."""
        try:
            from datasets import load_dataset
            
            dataset = load_dataset("imdb", split="train")
            texts = [example["text"] for example in dataset]
            labels = [example["label"] for example in dataset]
            
            return texts, labels
        except ImportError:
            return None, None
    
    def _try_load_from_csv(self, path: str) -> Tuple[List[str], List[int]]:
        """Try to load IMDB from CSV file."""
        if not os.path.exists(path):
            return None, None
        
        texts = []
        labels = []
        
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'text' in row and 'label' in row:
                    texts.append(row['text'])
                    labels.append(int(row['label']))
        
        if not texts:
            return None, None
        
        return texts, labels
    
    def load_imdb(
        self,
        source: str = "sample",  # "sample", "csv", or "huggingface"
        csv_path: Optional[str] = None,
    ) -> Tuple[DataLoader, DataLoader, Vocabulary]:
        """
        Load IMDB dataset and create data loaders.
        
        Args:
            source: Data source - "sample", "csv", or "huggingface"
            csv_path: Path to CSV file (if source="csv")
        
        Returns:
            Tuple of (train_loader, test_loader, vocab)
        """
        random.seed(self.seed)
        
        # Load data based on source
        if source == "csv" and csv_path:
            texts, labels = self._try_load_from_csv(csv_path)
            if texts is None:
                print(f"CSV not found at {csv_path}, using sample data")
                texts, labels = self._generate_sample_data()
        elif source == "huggingface":
            texts, labels = self._try_load_from_huggingface()
            if texts is None:
                print("HuggingFace datasets not available, using sample data")
                texts, labels = self._generate_sample_data()
        else:
            texts, labels = self._generate_sample_data()
        
        print(f"Loaded {len(texts)} IMDB samples")
        print(f"Class distribution: 0={labels.count(0)}, 1={labels.count(1)}")
        
        # Build vocabulary
        self.vocab = Vocabulary(min_freq=2, max_vocab_size=self.vocab_size)
        self.vocab.build_vocab(texts)
        print(f"Vocabulary size: {self.vocab.vocab_size}")
        
        # Split data
        combined = list(zip(texts, labels))
        random.shuffle(combined)
        
        split_idx = int(len(combined) * self.train_ratio)
        train_data = combined[:split_idx]
        test_data = combined[split_idx:]
        
        train_texts, train_labels = zip(*train_data)
        test_texts, test_labels = zip(*test_data)
        
        # Create datasets
        train_dataset = IMDBDataset(
            texts=list(train_texts),
            labels=list(train_labels),
            vocab=self.vocab,
            max_length=self.max_length,
        )
        
        test_dataset = IMDBDataset(
            texts=list(test_texts),
            labels=list(test_labels),
            vocab=self.vocab,
            max_length=self.max_length,
        )
        
        # Create data loaders
        collator = DataCollator(vocab=self.vocab)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collator,
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collator,
        )
        
        return self.train_loader, self.test_loader, self.vocab


def get_imdb_expected_results() -> IMDBExpectedResults:
    """Get expected results schema for IMDB dataset."""
    return IMDBExpectedResults()


# Sample usage
if __name__ == "__main__":
    print("=" * 60)
    print("IMDB Dataset Loader Demo")
    print("=" * 60)
    
    loader = IMDBDataLoader(vocab_size=5000, batch_size=8)
    train_loader, test_loader, vocab = loader.load_imdb()
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test a batch
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  labels: {batch['labels'].shape}")
    print(f"  attention_mask: {batch['attention_mask'].shape}")
    
    # Expected results
    expected = get_imdb_expected_results()
    print(f"\nExpected Results:")
    print(f"  Task type: {expected.task_type}")
    print(f"  Dataset: {expected.dataset_name}")
    print(f"  Classes: {expected.num_classes}")
    print(f"  Class names: {expected.class_names}")
