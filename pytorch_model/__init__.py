"""
Custom PyTorch Transformer Model Package.

A complete implementation of a transformer-based text classifier with:
- Custom multi-head attention mechanism
- Sinusoidal and learnable positional encodings
- Transformer encoder layers
- Classification head

Example usage:
    >>> from model import CustomTransformer
    >>> from config import ModelConfig
    >>> 
    >>> config = ModelConfig(vocab_size=10000, hidden_size=256)
    >>> model = CustomTransformer(config)
    >>> 
    >>> import torch
    >>> input_ids = torch.randint(0, 10000, (2, 32))
    >>> outputs = model(input_ids)
    >>> print(outputs['logits'].shape)  # torch.Size([2, 2])
"""

from .config import ModelConfig, TrainingConfig, DataConfig
from .model import CustomTransformer, create_model
from .attention import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    CausalSelfAttention,
    AttentionOutput,
)
from .embeddings import (
    TokenEmbedding,
    SinusoidalPositionalEncoding,
    LearnablePositionalEncoding,
    CombinedEmbedding,
)
from .dataset import (
    Vocabulary,
    TextClassificationDataset,
    DataCollator,
    generate_sample_data,
    split_data,
    create_simple_dataloader,
)
from .trainer import (
    Trainer,
    EarlyStopping,
    CosineAnnealingWarmupScheduler,
    create_optimizer,
    compute_metrics,
    format_metrics,
)

__version__ = "1.0.0"
__author__ = "Custom AI"

__all__ = [
    # Config
    "ModelConfig",
    "TrainingConfig", 
    "DataConfig",
    # Model
    "CustomTransformer",
    "create_model",
    # Attention
    "ScaledDotProductAttention",
    "MultiHeadAttention",
    "CausalSelfAttention",
    "AttentionOutput",
    # Embeddings
    "TokenEmbedding",
    "SinusoidalPositionalEncoding",
    "LearnablePositionalEncoding",
    "CombinedEmbedding",
    # Dataset
    "Vocabulary",
    "TextClassificationDataset",
    "DataCollator",
    "generate_sample_data",
    "split_data",
    "create_simple_dataloader",
    # Trainer
    "Trainer",
    "EarlyStopping",
    "CosineAnnealingWarmupScheduler",
    "create_optimizer",
    "compute_metrics",
    "format_metrics",
]
