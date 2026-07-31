"""
Custom Attention Mechanisms for Transformer Model.

This module implements various attention mechanisms including:
- Scaled Dot-Product Attention
- Multi-Head Attention
- Multi-Head Attention with relative positional bias
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.
    
    Computes attention weights as:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    
    Args:
        dropout_prob: Dropout probability for attention weights
    """
    
    def __init__(self, dropout_prob: float = 0.1):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()
    
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass of scaled dot-product attention.
        
        Args:
            query: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
            key: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
            value: Value tensor of shape (batch_size, num_heads, seq_len, head_dim)
            attention_mask: Optional mask tensor of shape (batch_size, 1, 1, seq_len)
            output_attentions: Whether to return attention weights
        
        Returns:
            output: Attention output of shape (batch_size, num_heads, seq_len, head_dim)
            attention_weights: Optional attention weights of shape 
                               (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, num_heads, seq_len, head_dim = query.shape
        
        # Compute attention scores: QK^T / sqrt(d_k)
        # Shape: (batch_size, num_heads, seq_len, seq_len)
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Mask should be broadcastable: (batch_size, 1, 1, seq_len)
            attention_scores = attention_scores + attention_mask
        
        # Apply softmax to get attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Compute weighted sum of values
        output = torch.matmul(attention_probs, value)
        
        if output_attentions:
            return output, attention_probs
        return output, None


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention layer.
    
    Projects queries, keys, and values into multiple attention heads,
    computes attention in parallel, and concatenates the results.
    
    Args:
        hidden_size: Dimension of hidden states
        num_attention_heads: Number of attention heads
        head_dim: Dimension of each attention head (if None, computed as hidden_size/num_heads)
        dropout_prob: Dropout probability
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        head_dim: Optional[int] = None,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        
        if head_dim is None:
            self.head_dim = hidden_size // num_attention_heads
        else:
            self.head_dim = head_dim
        
        assert self.hidden_size % self.num_attention_heads == 0, (
            f"hidden_size ({self.hidden_size}) must be divisible by "
            f"num_attention_heads ({self.num_attention_heads})"
        )
        
        self.all_head_size = self.num_attention_heads * self.head_dim
        
        # Linear projections for Q, K, V
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        # Output projection
        self.output = nn.Linear(self.all_head_size, hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_prob)
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout_prob)
    
    def transpose_for_scores(self, x: Tensor) -> Tensor:
        """
        Transpose tensor from (batch_size, seq_len, all_head_size) 
        to (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        return x.transpose(1, 2)
    
    def transpose_back(self, x: Tensor) -> Tensor:
        """
        Transpose tensor from (batch_size, num_heads, seq_len, head_dim) 
        to (batch_size, seq_len, all_head_size)
        """
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2)
        return x.contiguous().view(batch_size, seq_len, self.all_head_size)
    
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass of multi-head attention.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional mask tensor (can be 2D, 3D, or 4D)
            output_attentions: Whether to return attention weights
        
        Returns:
            output: Output tensor of shape (batch_size, seq_len, hidden_size)
            attention_weights: Optional attention weights
        """
        # Project hidden states to Q, K, V
        query = self.transpose_for_scores(self.query(hidden_states))
        key = self.transpose_for_scores(self.key(hidden_states))
        value = self.transpose_for_scores(self.value(hidden_states))
        
        # Process attention mask for proper broadcasting
        processed_mask = None
        if attention_mask is not None:
            # Handle 4D mask: (batch, 1, 1, seq_len) -> (batch, 1, 1, seq_len)
            # Handle 3D mask: (batch, 1, seq_len) -> (batch, 1, 1, seq_len)
            # Handle 2D mask: (batch, seq_len) -> (batch, 1, 1, seq_len)
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            # If 4D, keep as is
            processed_mask = attention_mask
        
        # Compute attention
        context, attention_probs = self.attention(
            query, key, value, processed_mask, output_attentions
        )
        
        # Transpose back and concatenate heads
        context = self.transpose_back(context)
        context = self.output(context)
        context = self.dropout(context)
        
        return context, attention_probs


class CausalSelfAttention(nn.Module):
    """
    Causal Self-Attention for autoregressive models.
    
    Applies a causal mask to prevent attending to future tokens.
    Useful for language modeling tasks.
    
    Args:
        hidden_size: Dimension of hidden states
        num_attention_heads: Number of attention heads
        dropout_prob: Dropout probability
        max_seq_length: Maximum sequence length for causal mask caching
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        dropout_prob: float = 0.1,
        max_seq_length: int = 512,
    ):
        super().__init__()
        self.max_seq_length = max_seq_length
        
        self.attention = MultiHeadAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            dropout_prob=dropout_prob,
        )
        
        # Causal mask buffer (created on first forward pass)
        self.register_buffer(
            "causal_mask", 
            self._create_causal_mask(max_seq_length),
            persistent=False
        )
    
    @staticmethod
    def _create_causal_mask(seq_length: int) -> Tensor:
        """Create a causal (lower triangular) mask."""
        mask = torch.triu(
            torch.ones(seq_length, seq_length, dtype=torch.bool), 
            diagonal=1
        )
        return mask
    
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass with causal masking.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional additional mask
            output_attentions: Whether to return attention weights
        
        Returns:
            output: Output tensor
            attention_weights: Optional attention weights
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Create or update causal mask for current sequence length
        if seq_len > self.causal_mask.shape[0]:
            self.causal_mask = self._create_causal_mask(seq_length=seq_length)
        
        causal_mask = self.causal_mask[:seq_len, :seq_len].unsqueeze(0).unsqueeze(0)
        causal_mask = causal_mask.to(hidden_states.device)
        
        # Combine with attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.masked_fill(causal_mask.squeeze(), float("-inf"))
        else:
            attention_mask = causal_mask * float("-inf")
        
        return self.attention(hidden_states, attention_mask, output_attentions)


class AttentionOutput(nn.Module):
    """
    Attention output layer with residual connection and layer normalization.
    
    Implements: LayerNorm(x + Attention(x))
    """
    
    def __init__(self, hidden_size: int, dropout_prob: float = 0.1, layer_norm_eps: float = 1e-12):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
    
    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        """Apply attention output transformation."""
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states
