"""
Configuration settings for the Custom Transformer Model.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration class for model hyperparameters."""
    
    # Model architecture
    vocab_size: int = 30522
    hidden_size: int = 256
    num_attention_heads: int = 8
    num_hidden_layers: int = 4
    intermediate_size: int = 512
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    
    # Classification
    num_labels: int = 2
    max_position_embeddings: int = 512
    
    # Activation
    hidden_act: str = "gelu"
    
    # Layer normalization
    layer_norm_eps: float = 1e-12
    
    # Initializer range for weights
    initializer_range: float = 0.02
    
    # Pooling
    pooling_type: str = "mean"  # "mean" or "cls"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.hidden_size % self.num_attention_heads == 0, (
            f"hidden_size ({self.hidden_size}) must be divisible by "
            f"num_attention_heads ({self.num_attention_heads})"
        )
        assert self.hidden_act in ["gelu", "relu", "swish"], (
            f"hidden_act must be one of ['gelu', 'relu', 'swish'], got {self.hidden_act}"
        )
        assert self.pooling_type in ["mean", "cls"], (
            f"pooling_type must be 'mean' or 'cls', got {self.pooling_type}"
        )
    
    @property
    def attention_head_size(self) -> int:
        """Size of each attention head."""
        return self.hidden_size // self.num_attention_heads


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""
    
    # Optimization
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    beta1: float = 0.9
    beta2: float = 0.999
    
    # Training loop
    num_epochs: int = 10
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Scheduler
    warmup_steps: int = 0
    warmup_ratio: float = 0.1
    
    # Early stopping
    patience: int = 3
    min_delta: float = 1e-4
    
    # Logging
    logging_steps: int = 100
    save_steps: int = 1000
    
    # Reproducibility
    seed: int = 42
    
    # Mixed precision
    use_amp: bool = False
    
    # Device
    device: str = "cuda"


@dataclass
class DataConfig:
    """Configuration for data processing."""
    
    max_seq_length: int = 128
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Data augmentation
    use_data_augmentation: bool = False
    
    # Tokenization
    padding: str = "max_length"
    truncation: bool = True
    return_tensors: str = "pt"
