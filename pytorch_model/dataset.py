"""
Dataset and Data Loading utilities for the Transformer Model.

This module provides:
- Custom Dataset classes for text classification
- Data collation and padding
- Simple vocabulary and tokenization
"""

import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Sampler


class Vocabulary:
    """
    Simple vocabulary class for text tokenization.
    
    Maps tokens to indices and vice versa.
    Special tokens: [PAD]=0, [UNK]=1, [CLS]=2, [SEP]=3, [MASK]=4
    
    Example:
        >>> vocab = Vocabulary(min_freq=2)
        >>> vocab.build_vocab(["hello world", "hello"])
        >>> ids = vocab.encode("hello world")
        >>> tokens = vocab.decode(ids)
    """
    
    def __init__(
        self,
        min_freq: int = 1,
        max_vocab_size: Optional[int] = None,
        special_tokens: Optional[List[str]] = None,
    ):
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        
        # Default special tokens
        self.special_tokens = special_tokens or ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        
        self._token_to_idx: Dict[str, int] = {}
        self._idx_to_token: Dict[int, str] = {}
        self._token_counts: Counter = Counter()
        
        self._build_special_tokens()
    
    def _build_special_tokens(self):
        """Add special tokens to vocabulary."""
        for idx, token in enumerate(self.special_tokens):
            self._token_to_idx[token] = idx
            self._idx_to_token[idx] = token
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self._token_to_idx)
    
    @property
    def pad_idx(self) -> int:
        """Return padding token index."""
        return self._token_to_idx.get("[PAD]", 0)
    
    @property
    def unk_idx(self) -> int:
        """Return unknown token index."""
        return self._token_to_idx.get("[UNK]", 1)
    
    @property
    def cls_idx(self) -> int:
        """Return classification token index."""
        return self._token_to_idx.get("[CLS]", 2)
    
    @property
    def sep_idx(self) -> int:
        """Return separator token index."""
        return self._token_to_idx.get("[SEP]", 3)
    
    def build_vocab(self, texts: List[str]):
        """
        Build vocabulary from list of texts.
        
        Args:
            texts: List of text strings
        """
        # Count token frequencies
        for text in texts:
            tokens = self._tokenize(text)
            self._token_counts.update(tokens)
        
        # Add tokens that meet minimum frequency
        current_idx = len(self.special_tokens)
        for token, count in self._token_counts.most_common(self.max_vocab_size):
            if count >= self.min_freq:
                if token not in self._token_to_idx:
                    self._token_to_idx[token] = current_idx
                    self._idx_to_token[current_idx] = token
                    current_idx += 1
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization."""
        return text.lower().split()
    
    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        add_special_tokens: bool = True,
    ) -> List[int]:
        """
        Convert text to token IDs.
        
        Args:
            text: Input text string
            max_length: Maximum sequence length
            add_special_tokens: Whether to add [CLS] and [SEP] tokens
        
        Returns:
            List of token IDs
        """
        tokens = self._tokenize(text)
        token_ids = [
            self._token_to_idx.get(token, self.unk_idx)
            for token in tokens
        ]
        
        if add_special_tokens:
            token_ids = [self.cls_idx] + token_ids + [self.sep_idx]
        
        if max_length is not None:
            token_ids = self._pad_or_truncate(token_ids, max_length, add_special_tokens)
        
        return token_ids
    
    def _pad_or_truncate(
        self,
        token_ids: List[int],
        max_length: int,
        add_special_tokens: bool,
    ) -> List[int]:
        """Truncate or pad sequence to max_length."""
        sep_len = 2 if add_special_tokens else 0
        effective_max = max_length - sep_len
        
        if len(token_ids) > max_length:
            # Truncate
            if add_special_tokens:
                return token_ids[:max_length - 1] + [self.sep_idx]
            return token_ids[:max_length]
        else:
            # Pad
            padding_length = max_length - len(token_ids)
            return token_ids + [self.pad_idx] * padding_length
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Convert token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output
        
        Returns:
            Decoded text string
        """
        tokens = []
        for idx in token_ids:
            token = self._idx_to_token.get(idx, "[UNK]")
            if skip_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token)
        
        # Clean up
        text = " ".join(tokens)
        text = text.replace("[UNK]", "?")
        return text
    
    def __len__(self) -> int:
        return self.vocab_size


class TextClassificationDataset(Dataset):
    """
    Dataset for text classification tasks.
    
    Args:
        texts: List of input texts
        labels: List of corresponding labels
        vocab: Vocabulary object for tokenization
        max_length: Maximum sequence length
        transform: Optional transform to apply to texts
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        vocab: Vocabulary,
        max_length: int = 128,
        transform: Optional[callable] = None,
    ):
        assert len(texts) == len(labels), (
            f"Number of texts ({len(texts)}) must match number of labels ({len(labels)})"
        )
        
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Get a single item from the dataset.
        
        Returns:
            Tuple of (input_ids, label)
        """
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Apply transform if provided
        if self.transform:
            text = self.transform(text)
        
        # Encode text
        token_ids = self.vocab.encode(text, max_length=self.max_length)
        
        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )
    
    def get_raw_item(self, idx: int) -> Tuple[str, int]:
        """Get raw text and label without tokenization."""
        return self.texts[idx], self.labels[idx]


class DataCollator:
    """
    Collate function for batching and padding sequences.
    
    Args:
        vocab: Vocabulary for padding token index
        pad_token_id: ID of the padding token
        return_attention_mask: Whether to return attention masks
    """
    
    def __init__(
        self,
        vocab: Optional[Vocabulary] = None,
        pad_token_id: int = 0,
        return_attention_mask: bool = True,
    ):
        self.vocab = vocab
        self.pad_token_id = pad_token_id
        self.return_attention_mask = return_attention_mask
    
    def __call__(self, batch: List[Tuple[Tensor, Tensor]]) -> Dict[str, Tensor]:
        """
        Collate a batch of samples.
        
        Args:
            batch: List of (input_ids, label) tuples
        
        Returns:
            Dictionary with batched tensors
        """
        input_ids, labels = zip(*batch)
        
        # Stack and pad input_ids
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        
        output = {
            "input_ids": input_ids,
            "labels": labels,
        }
        
        # Create attention mask (1 for real tokens, 0 for padding)
        if self.return_attention_mask:
            attention_mask = (input_ids != self.pad_token_id).long()
            output["attention_mask"] = attention_mask
        
        return output


def create_simple_dataloader(
    texts: List[str],
    labels: List[int],
    vocab: Vocabulary,
    batch_size: int = 32,
    max_length: int = 128,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Convenience function to create a DataLoader.
    
    Args:
        texts: List of input texts
        labels: List of labels
        vocab: Vocabulary object
        batch_size: Batch size
        max_length: Maximum sequence length
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
    
    Returns:
        DataLoader instance
    """
    dataset = TextClassificationDataset(
        texts=texts,
        labels=labels,
        vocab=vocab,
        max_length=max_length,
    )
    
    collator = DataCollator(vocab=vocab)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
    )


def generate_sample_data(
    num_samples: int = 1000,
    num_classes: int = 2,
    avg_text_length: int = 20,
    seed: int = 42,
) -> Tuple[List[str], List[int]]:
    """
    Generate synthetic text classification data for testing.
    
    Args:
        num_samples: Number of samples to generate
        num_classes: Number of classification classes
        avg_text_length: Average length of texts (in words)
        seed: Random seed
    
    Returns:
        Tuple of (texts, labels)
    """
    random.seed(seed)
    
    # Define some topic keywords for each class
    topics = {
        0: ["science", "research", "experiment", "hypothesis", "data", "study", 
            "analysis", "theory", "discovery", "laboratory"],
        1: ["sports", "game", "team", "player", "match", "championship", 
            "score", "league", "tournament", "victory"],
    }
    
    if num_classes > 2:
        topics[2] = ["business", "market", "company", "investment", "stock", 
                    "economy", "finance", "profit", "growth", "trade"]
    if num_classes > 3:
        topics[3] = ["entertainment", "movie", "music", "celebrity", "film",
                    "show", "actor", "director", "album", "concert"]
    
    texts = []
    labels = []
    
    for _ in range(num_samples):
        # Assign random class
        label = random.randint(0, num_classes - 1)
        
        # Generate text with topic keywords
        num_words = max(5, int(random.gauss(avg_text_length, 5)))
        words = random.choices(topics[label], k=num_words)
        
        # Add some filler words
        fillers = ["the", "a", "is", "was", "are", "have", "has", "will", 
                   "can", "could", "would", "should", "very", "really"]
        all_words = words + random.choices(fillers, k=len(words) // 2)
        random.shuffle(all_words)
        
        text = " ".join(all_words)
        texts.append(text)
        labels.append(label)
    
    return texts, labels


def split_data(
    texts: List[str],
    labels: List[int],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        texts: List of texts
        labels: List of labels
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        seed: Random seed
    
    Returns:
        Tuple of (train_texts, train_labels, val_texts, val_labels, test_texts, test_labels)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "Ratios must sum to 1.0"
    )
    
    random.seed(seed)
    
    # Create combined list and shuffle
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    
    # Split
    n = len(combined)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_data = combined[:train_end]
    val_data = combined[train_end:val_end]
    test_data = combined[val_end:]
    
    # Unpack
    train_texts, train_labels = zip(*train_data) if train_data else ([], [])
    val_texts, val_labels = zip(*val_data) if val_data else ([], [])
    test_texts, test_labels = zip(*test_data) if test_data else ([], [])
    
    return (
        list(train_texts), list(train_labels),
        list(val_texts), list(val_labels),
        list(test_texts), list(test_labels),
    )
