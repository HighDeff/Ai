"""
Custom Transformer Model for Text Classification.

This module implements a complete transformer-based classifier with:
- Custom multi-head attention
- Transformer encoder layers
- Token and positional embeddings
- Classification head
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from config import ModelConfig
from attention import MultiHeadAttention, AttentionOutput
from embeddings import CombinedEmbedding


class GELUActivation(nn.Module):
    """
    Gaussian Error Linear Unit (GELU) activation function.
    
    GELU(x) = x * Phi(x) where Phi is the cumulative distribution function
    of the standard normal distribution.
    
    Approximation used: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        return F.gelu(input)


class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    Implements a two-layer linear transformation with activation:
        FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    
    Args:
        hidden_size: Input and output dimension
        intermediate_size: Hidden layer dimension (typically 4x hidden_size)
        dropout_prob: Dropout probability
        hidden_act: Activation function ('gelu', 'relu', or 'swish')
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout_prob: float = 0.1,
        hidden_act: str = "gelu",
    ):
        super().__init__()
        
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        
        if hidden_act == "gelu":
            self.intermediate_act_fn = GELUActivation()
        elif hidden_act == "relu":
            self.intermediate_act_fn = nn.ReLU()
        elif hidden_act == "swish":
            self.intermediate_act_fn = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {hidden_act}")
        
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
    
    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        """Apply feed-forward network with residual connection."""
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer.
    
    Combines multi-head self-attention with a feed-forward network,
    using pre-normalization architecture with residual connections.
    
    Architecture:
        LN -> Attention -> Add -> LN -> FFN -> Add
    
    Args:
        config: Model configuration object
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.attention = MultiHeadAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            dropout_prob=config.attention_dropout_prob,
        )
        self.attention_output = AttentionOutput(
            hidden_size=config.hidden_size,
            dropout_prob=config.hidden_dropout_prob,
            layer_norm_eps=config.layer_norm_eps,
        )
        
        self.ffn = FeedForwardNetwork(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            dropout_prob=config.hidden_dropout_prob,
            hidden_act=config.hidden_act,
        )
        
        # Pre-norm layer normalization
        self.layer_norm1 = nn.LayerNorm(
            config.hidden_size, 
            eps=config.layer_norm_eps
        )
        self.layer_norm2 = nn.LayerNorm(
            config.hidden_size, 
            eps=config.layer_norm_eps
        )
    
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass through encoder layer.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_length, hidden_size)
            attention_mask: Optional attention mask
            output_attentions: Whether to return attention weights
        
        Returns:
            hidden_states: Output tensor
            attention_weights: Optional attention weights
        """
        # Pre-norm: Layer Normalization before attention
        layernorm1_input = self.layer_norm1(hidden_states)
        
        # Self-attention with residual connection
        attention_output, attention_weights = self.attention(
            layernorm1_input,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + attention_output
        
        # Pre-norm: Layer Normalization before FFN
        layernorm2_input = self.layer_norm2(hidden_states)
        
        # Feed-forward network with residual connection
        ffn_output = self.ffn(layernorm2_input, torch.zeros_like(layernorm2_input))
        hidden_states = hidden_states + ffn_output
        
        return hidden_states, attention_weights


class TransformerEncoder(nn.Module):
    """
    Stack of Transformer Encoder Layers.
    
    Args:
        config: Model configuration object
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(config) 
            for _ in range(config.num_hidden_layers)
        ])
        
        self.num_layers = config.num_hidden_layers
    
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        output_all_attentions: bool = False,
    ) -> Tuple[Tensor, Optional[List[Tensor]]]:
        """
        Forward pass through all encoder layers.
        
        Args:
            hidden_states: Input tensor
            attention_mask: Optional attention mask
            output_all_attentions: Whether to return attention from all layers
        
        Returns:
            hidden_states: Final hidden states
            all_attentions: Optional list of attention weights from all layers
        """
        all_attentions = [] if output_all_attentions else None
        
        for layer in self.layers:
            hidden_states, attention_weights = layer(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_all_attentions,
            )
            
            if output_all_attentions and attention_weights is not None:
                all_attentions.append(attention_weights)
        
        return hidden_states, all_attentions


class ClassificationHead(nn.Module):
    """
    Classification head for text classification tasks.
    
    Applies pooling, dropout, and a linear layer to produce class logits.
    
    Args:
        hidden_size: Dimension of hidden states
        num_labels: Number of classification labels
        dropout_prob: Dropout probability
        pooling_type: Type of pooling ('mean' or 'cls')
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        dropout_prob: float = 0.1,
        pooling_type: str = "mean",
    ):
        super().__init__()
        
        assert pooling_type in ["mean", "cls"], (
            f"pooling_type must be 'mean' or 'cls', got {pooling_type}"
        )
        
        self.pooling_type = pooling_type
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)
    
    def pooled_forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """
        Apply pooling to get a single vector representation.
        
        Args:
            hidden_states: Hidden states of shape (batch_size, seq_length, hidden_size)
            attention_mask: Optional attention mask
        
        Returns:
            pooled_output: Pooled tensor of shape (batch_size, hidden_size)
        """
        if self.pooling_type == "cls":
            # Use [CLS] token representation (first token)
            pooled_output = hidden_states[:, 0]
        else:
            # Mean pooling over sequence
            if attention_mask is not None:
                # Handle different attention mask dimensions
                # Convert to 2D: (batch, seq_len)
                if attention_mask.dim() == 4:
                    attention_mask = attention_mask.squeeze(1).squeeze(1)
                elif attention_mask.dim() == 3:
                    attention_mask = attention_mask.squeeze(1)
                
                # Convert from -inf mask to 0/1 mask
                mask = (attention_mask > -1e9).float()
                
                # Expand mask to match hidden_states: (batch, seq_len) -> (batch, seq_len, 1)
                mask = mask.unsqueeze(-1)
                
                # Apply mask and sum
                sum_embeddings = torch.sum(hidden_states * mask, dim=1)
                sum_mask = mask.sum(dim=1)
                sum_mask = torch.clamp(sum_mask, min=1e-9)
                pooled_output = sum_embeddings / sum_mask
            else:
                pooled_output = torch.mean(hidden_states, dim=1)
        
        return pooled_output
    
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass through classification head.
        
        Args:
            hidden_states: Hidden states from encoder
            attention_mask: Optional attention mask
        
        Returns:
            logits: Class logits of shape (batch_size, num_labels)
        """
        pooled_output = self.pooled_forward(hidden_states, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class CustomTransformer(nn.Module):
    """
    Custom Transformer Model for Text Classification.
    
    A complete transformer architecture implementing:
    - Token and positional embeddings
    - Stack of transformer encoder layers
    - Classification head with pooling
    
    Example usage:
        >>> config = ModelConfig(vocab_size=1000, hidden_size=128, num_labels=2)
        >>> model = CustomTransformer(config)
        >>> input_ids = torch.randint(0, 1000, (2, 32))  # batch_size=2, seq_len=32
        >>> logits = model(input_ids)
        >>> print(logits.shape)  # torch.Size([2, 2])
    
    Args:
        config: Model configuration object
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.config = config
        
        # Embeddings
        self.embeddings = CombinedEmbedding(
            vocab_size=config.vocab_size,
            embedding_dim=config.hidden_size,
            max_seq_length=config.max_position_embeddings,
            positional_encoding_type="sinusoidal",
            dropout_prob=config.hidden_dropout_prob,
            padding_idx=0,
            layer_norm_eps=config.layer_norm_eps,
            init_range=config.initializer_range,
        )
        
        # Encoder
        self.encoder = TransformerEncoder(config)
        
        # Classification head
        self.classifier = ClassificationHead(
            hidden_size=config.hidden_size,
            num_labels=config.num_labels,
            dropout_prob=config.hidden_dropout_prob,
            pooling_type=config.pooling_type,
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights using Xavier/Glorot initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Get the weight attribute (could be 'weight' for nn.Embedding)
            weight = getattr(module, 'weight', None)
            if weight is not None:
                nn.init.trunc_normal_(
                    weight, 
                    mean=0.0, 
                    std=self.config.initializer_range
                )
                if module.padding_idx is not None:
                    nn.init.zeros_(weight[module.padding_idx])
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def get_num_params(self, trainable_only: bool = False) -> int:
        """
        Count the number of parameters in the model.
        
        Args:
            trainable_only: If True, count only trainable parameters
        
        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def get_embedding_output(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Get embedding output without going through encoder."""
        return self.embeddings(input_ids)
    
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Dict[str, Any]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_length)
            attention_mask: Optional attention mask
            labels: Optional labels for loss computation
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
        
        Returns:
            Dictionary containing:
                - logits: Class logits
                - loss: Optional cross-entropy loss
                - attentions: Optional attention weights
                - hidden_states: Optional hidden states
        """
        # Get embeddings
        embedding_output = self.embeddings(input_ids)
        
        # Get attention mask if not provided
        if attention_mask is None:
            attention_mask = self.embeddings.create_attention_mask(input_ids)
        
        # Encode
        encoder_output, all_attentions = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            output_all_attentions=output_attentions,
        )
        
        # Classify
        logits = self.classifier(encoder_output, attention_mask)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        # Prepare output
        output = {
            "logits": logits,
            "loss": loss,
        }
        
        if output_attentions:
            output["attentions"] = all_attentions
        
        if output_hidden_states:
            output["hidden_states"] = (embedding_output, encoder_output)
        
        return output
    
    def predict(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """
        Make predictions on input.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
        
        Returns:
            predictions: Predicted class indices
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            predictions = torch.argmax(outputs["logits"], dim=-1)
        return predictions
    
    def predict_proba(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """
        Get prediction probabilities.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
        
        Returns:
            probabilities: Class probabilities (softmax over logits)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            probabilities = F.softmax(outputs["logits"], dim=-1)
        return probabilities
    
    def save_pretrained(self, path: str):
        """Save model to path."""
        torch.save({
            "config": self.config,
            "state_dict": self.state_dict(),
        }, path)
    
    @classmethod
    def from_pretrained(cls, path: str) -> "CustomTransformer":
        """Load model from path."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model


def create_model(
    vocab_size: int = 30522,
    hidden_size: int = 256,
    num_attention_heads: int = 8,
    num_hidden_layers: int = 4,
    num_labels: int = 2,
    intermediate_size: Optional[int] = None,
    dropout: float = 0.1,
) -> CustomTransformer:
    """
    Convenience function to create a model with specified parameters.
    
    Args:
        vocab_size: Vocabulary size
        hidden_size: Hidden dimension
        num_attention_heads: Number of attention heads
        num_hidden_layers: Number of encoder layers
        num_labels: Number of classification labels
        intermediate_size: FFN intermediate size (default: 4 * hidden_size)
        dropout: Dropout probability
    
    Returns:
        CustomTransformer model
    """
    config = ModelConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=intermediate_size or hidden_size * 4,
        num_labels=num_labels,
        hidden_dropout_prob=dropout,
        attention_dropout_prob=dropout,
    )
    return cls(config)
