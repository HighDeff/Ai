"""
Embedding Layers for Transformer Model.

This module implements:
- Token Embeddings
- Positional Encodings (Sinusoidal and Learnable)
- Combined Embeddings with dropout
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TokenEmbedding(nn.Module):
    """
    Token embedding layer that maps token IDs to dense vectors.
    
    Args:
        vocab_size: Size of the vocabulary
        embedding_dim: Dimension of the embedding vectors
        padding_idx: Optional index for padding token
        init_range: Range for weight initialization
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        init_range: float = 0.02,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        
        self._init_weights(init_range)
    
    def _init_weights(self, init_range: float):
        """Initialize embedding weights."""
        nn.init.trunc_normal_(self.embedding.weight, mean=0.0, std=init_range)
        
        if self.padding_idx is not None:
            nn.init.zeros_(self.embedding.weight[self.padding_idx])
    
    def forward(self, input_ids: Tensor) -> Tensor:
        """Get token embeddings for input IDs."""
        return self.embedding(input_ids)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as described in "Attention Is All You Need".
    
    Uses sine and cosine functions of different frequencies to encode positions:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    This encoding provides a way for the model to attend to relative positions,
    as the patterns are unique for each position.
    
    Args:
        embedding_dim: Dimension of the embeddings
        max_seq_length: Maximum sequence length to pre-compute encodings for
    """
    
    def __init__(self, embedding_dim: int, max_seq_length: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Create position encoding matrix
        pe = self._create_encoding(max_seq_length, embedding_dim)
        # Register as buffer (not a parameter, but should be moved with the model)
        self.register_buffer("pe", pe)
    
    @staticmethod
    def _create_encoding(seq_length: int, embedding_dim: int) -> Tensor:
        """Create sinusoidal positional encoding matrix."""
        # Create position indices
        position = torch.arange(seq_length).unsqueeze(1)
        
        # Create frequency divisions
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim)
        )
        
        # Compute sine for even indices and cosine for odd indices
        pe = torch.zeros(seq_length, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: (1, seq_length, embedding_dim)
        return pe.unsqueeze(0)
    
    def forward(self, seq_length: int, device: torch.device) -> Tensor:
        """
        Get positional encoding for a given sequence length.
        
        Args:
            seq_length: Length of the input sequence
            device: Device to create tensor on
        
        Returns:
            Positional encoding tensor of shape (1, seq_length, embedding_dim)
        """
        return self.pe[:, :seq_length, :].to(device)


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable Positional Encoding.
    
    Instead of using fixed sinusoidal patterns, learns positional embeddings
    during training. More flexible but may require more data to generalize.
    
    Args:
        max_seq_length: Maximum sequence length
        embedding_dim: Dimension of the embeddings
        init_range: Range for weight initialization
    """
    
    def __init__(
        self,
        max_seq_length: int,
        embedding_dim: int,
        init_range: float = 0.02,
    ):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        
        self.embedding = nn.Embedding(max_seq_length, embedding_dim)
        self._init_weights(init_range)
    
    def _init_weights(self, init_range: float):
        """Initialize positional embedding weights."""
        nn.init.trunc_normal_(self.embedding.weight, mean=0.0, std=init_range)
    
    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Get positional embeddings based on sequence positions.
        
        Args:
            input_ids: Input tensor of shape (batch_size, seq_length)
        
        Returns:
            Positional embeddings of shape (batch_size, seq_length, embedding_dim)
        """
        batch_size, seq_length = input_ids.shape
        
        # Create position indices: (batch_size, seq_length)
        positions = torch.arange(seq_length, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        
        return self.embedding(positions)


class CombinedEmbedding(nn.Module):
    """
    Combined embedding layer that adds token embeddings and positional encodings.
    
    Supports both sinusoidal (fixed) and learnable positional encodings.
    
    Args:
        vocab_size: Size of the vocabulary
        embedding_dim: Dimension of the embeddings
        max_seq_length: Maximum sequence length
        positional_encoding_type: Type of positional encoding ('sinusoidal' or 'learnable')
        dropout_prob: Dropout probability
        padding_idx: Index of padding token
        layer_norm_eps: Epsilon for layer normalization
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        max_seq_length: int = 512,
        positional_encoding_type: str = "sinusoidal",
        dropout_prob: float = 0.1,
        padding_idx: Optional[int] = 0,
        layer_norm_eps: float = 1e-12,
        init_range: float = 0.02,
    ):
        super().__init__()
        
        assert positional_encoding_type in ["sinusoidal", "learnable"], (
            f"positional_encoding_type must be 'sinusoidal' or 'learnable', "
            f"got {positional_encoding_type}"
        )
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.positional_encoding_type = positional_encoding_type
        
        # Token embeddings
        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            init_range=init_range,
        )
        
        # Positional encodings
        if positional_encoding_type == "sinusoidal":
            self.positional_encoding = SinusoidalPositionalEncoding(
                embedding_dim=embedding_dim,
                max_seq_length=max_seq_length,
            )
        else:
            self.positional_encoding = LearnablePositionalEncoding(
                max_seq_length=max_seq_length,
                embedding_dim=embedding_dim,
                init_range=init_range,
            )
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_prob)
        
        self.padding_idx = padding_idx
    
    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Compute combined token and positional embeddings.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_length)
        
        Returns:
            Combined embeddings of shape (batch_size, seq_length, embedding_dim)
        """
        # Get token embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Get positional encodings
        batch_size, seq_length = input_ids.shape
        if self.positional_encoding_type == "sinusoidal":
            pos_embeds = self.positional_encoding(seq_length, input_ids.device)
        else:
            pos_embeds = self.positional_encoding(input_ids)
        
        # Combine embeddings
        embeddings = token_embeds + pos_embeds
        
        # Apply layer norm and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
    def create_attention_mask(self, input_ids: Tensor) -> Optional[Tensor]:
        """
        Create attention mask from input IDs.
        
        Args:
            input_ids: Input token IDs
        
        Returns:
            Attention mask where 1 indicates valid tokens and 0 indicates padding
        """
        if self.padding_idx is None:
            return None
        
        # Create mask: 1 for real tokens, 0 for padding
        attention_mask = (input_ids != self.padding_idx).long()
        
        # Expand dimensions for broadcasting: (batch_size, 1, 1, seq_length)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        # Convert to format suitable for attention: 0 = valid, -inf = masked
        attention_mask = (1.0 - attention_mask) * -10000.0
        
        return attention_mask


def get_positional_encoding(
    seq_length: int,
    embedding_dim: int,
    device: torch.device,
    encoding_type: str = "sinusoidal",
) -> Tensor:
    """
    Standalone function to compute positional encodings.
    
    Args:
        seq_length: Length of the sequence
        embedding_dim: Dimension of the embeddings
        device: Device to create tensor on
        encoding_type: Type of encoding ('sinusoidal' or 'learnable')
    
    Returns:
        Positional encoding tensor
    """
    if encoding_type == "sinusoidal":
        pe = torch.zeros(1, seq_length, embedding_dim)
        position = torch.arange(0, seq_length).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float() 
            * (-math.log(10000.0) / embedding_dim)
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe.to(device)
    else:
        positions = torch.arange(seq_length).unsqueeze(1).to(device)
        embeddings = positions / (torch.maximum(
            torch.tensor(1.0), 
            torch.arange(0, embedding_dim, 2).float().to(device) / embedding_dim
        ) ** (2 * torch.arange(0, embedding_dim // 2).float().to(device) / embedding_dim))
        return embeddings.unsqueeze(0)
