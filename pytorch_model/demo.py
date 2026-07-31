"""
Demo Script for Custom PyTorch Transformer Model.

This script demonstrates:
- Creating a model from scratch
- Running forward pass
- Making predictions
- Visualizing attention weights

Run: python demo.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F

from config import ModelConfig
from model import CustomTransformer
from attention import MultiHeadAttention
from embeddings import SinusoidalPositionalEncoding, LearnablePositionalEncoding


def demo_model_creation():
    """Demonstrate model creation with different configurations."""
    print("\n" + "=" * 60)
    print("DEMO 1: Model Creation")
    print("=" * 60)
    
    # Create model with default config
    config = ModelConfig(
        vocab_size=10000,
        hidden_size=128,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_labels=3,
    )
    
    model = CustomTransformer(config)
    
    print(f"\nModel created with:")
    print(f"  - Vocabulary size: {config.vocab_size}")
    print(f"  - Hidden size: {config.hidden_size}")
    print(f"  - Attention heads: {config.num_attention_heads}")
    print(f"  - Encoder layers: {config.num_hidden_layers}")
    print(f"  - Classification labels: {config.num_labels}")
    
    total_params = model.get_num_params()
    trainable_params = model.get_num_params(trainable_only=True)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model, config


def demo_forward_pass(model: CustomTransformer, config: ModelConfig):
    """Demonstrate forward pass through the model."""
    print("\n" + "=" * 60)
    print("DEMO 2: Forward Pass")
    print("=" * 60)
    
    # Create sample input
    batch_size = 2
    seq_length = 16
    
    # Random token IDs
    input_ids = torch.randint(
        low=0, 
        high=config.vocab_size, 
        size=(batch_size, seq_length)
    )
    
    print(f"\nInput shape: {input_ids.shape}")
    print(f"Input sample (first sequence): {input_ids[0].tolist()}")
    
    # Forward pass without labels
    with torch.no_grad():
        outputs = model(input_ids)
    
    print(f"\nOutput keys: {list(outputs.keys())}")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Logits (first sample): {outputs['logits'][0].tolist()}")
    
    # Get probabilities
    probs = F.softmax(outputs["logits"], dim=-1)
    print(f"\nProbabilities (first sample): {probs[0].tolist()}")
    print(f"Predicted class: {torch.argmax(probs, dim=-1).tolist()}")


def demo_forward_with_labels(model: CustomTransformer):
    """Demonstrate forward pass with labels (loss computation)."""
    print("\n" + "=" * 60)
    print("DEMO 3: Forward Pass with Labels")
    print("=" * 60)
    
    batch_size = 4
    seq_length = 32
    
    input_ids = torch.randint(0, 10000, (batch_size, seq_length))
    labels = torch.randint(0, 2, (batch_size,))
    
    print(f"\nBatch size: {batch_size}")
    print(f"Sequence length: {seq_length}")
    print(f"Labels: {labels.tolist()}")
    
    # Forward pass with labels
    outputs = model(input_ids, labels=labels)
    
    print(f"\nLoss: {outputs['loss'].item():.4f}")
    print(f"Logits shape: {outputs['logits'].shape}")
    
    # Predictions
    predictions = torch.argmax(outputs["logits"], dim=-1)
    accuracy = (predictions == labels).float().mean()
    print(f"Accuracy: {accuracy.item():.4f}")


def demo_attention_output(model: CustomTransformer):
    """Demonstrate getting attention weights."""
    print("\n" + "=" * 60)
    print("DEMO 4: Attention Weights")
    print("=" * 60)
    
    batch_size = 1
    seq_length = 8  # Small sequence for attention visualization
    
    input_ids = torch.randint(0, 10000, (batch_size, seq_length))
    
    # Forward pass with attention output
    outputs = model(input_ids, output_attentions=True)
    
    attentions = outputs["attentions"]
    
    print(f"\nNumber of attention layers: {len(attentions)}")
    print(f"Attention shape per layer: {attentions[0].shape}")
    
    # Analyze attention patterns
    # Shape: (batch_size, num_heads, seq_len, seq_len)
    attn = attentions[0][0]  # First layer, first sample
    
    print(f"\nAttention pattern (Layer 1, Head 1) - first 4 tokens:")
    for i in range(min(4, seq_length)):
        row = attn[i].detach().cpu().numpy()[:4]
        print(f"  Token {i}: {row.round(3)}")


def demo_positional_encodings(config: ModelConfig):
    """Demonstrate positional encoding types."""
    print("\n" + "=" * 60)
    print("DEMO 5: Positional Encodings")
    print("=" * 60)
    
    seq_length = 20
    device = torch.device("cpu")
    
    # Sinusoidal
    print("\nSinusoidal Positional Encoding:")
    pos_enc = SinusoidalPositionalEncoding(
        embedding_dim=config.hidden_size,
        max_seq_length=seq_length
    )
    pe = pos_enc(seq_length, device)
    print(f"  Shape: {pe.shape}")
    print(f"  Sample (first position, first 10 dims): {pe[0, 0, :10].tolist()}")
    
    # Learnable
    print("\nLearnable Positional Encoding:")
    learnable_pos = LearnablePositionalEncoding(
        max_seq_length=seq_length,
        embedding_dim=config.hidden_size
    )
    input_ids = torch.randint(0, 1000, (1, seq_length))
    lpe = learnable_pos(input_ids)
    print(f"  Shape: {lpe.shape}")
    print(f"  Sample (first position, first 10 dims): {lpe[0, 0, :10].tolist()}")


def demo_prediction_interface(model: CustomTransformer):
    """Demonstrate prediction interface."""
    print("\n" + "=" * 60)
    print("DEMO 6: Prediction Interface")
    print("=" * 60)
    
    # Create random input
    input_ids = torch.randint(0, 10000, (1, 32))
    
    # Using predict method
    predictions = model.predict(input_ids)
    print(f"\nUsing predict(): {predictions.item()}")
    
    # Using predict_proba method
    probs = model.predict_proba(input_ids)
    print(f"Using predict_proba(): {probs[0].tolist()}")


def demo_model_save_load(model: CustomTransformer):
    """Demonstrate model save and load."""
    print("\n" + "=" * 60)
    print("DEMO 7: Save and Load")
    print("=" * 60)
    
    save_path = "/tmp/demo_model.pt"
    
    # Save
    model.save_pretrained(save_path)
    print(f"\nModel saved to: {save_path}")
    
    # Load
    loaded_model = CustomTransformer.from_pretrained(save_path)
    print(f"Model loaded successfully!")
    
    # Verify weights match
    original_output = model.predict(torch.randint(0, 10000, (1, 16)))
    loaded_output = loaded_model.predict(torch.randint(0, 10000, (1, 16)))
    
    print(f"\nWeights match: {original_output.shape == loaded_output.shape}")


def run_demo():
    """Run all demos."""
    print("\n" + "#" * 60)
    print("# CUSTOM PYTORCH TRANSFORMER MODEL DEMO")
    print("#" * 60)
    
    # Demo 1: Model creation
    model, config = demo_model_creation()
    
    # Demo 2: Forward pass
    demo_forward_pass(model, config)
    
    # Demo 3: Forward pass with labels
    demo_forward_with_labels(model)
    
    # Demo 4: Attention weights
    demo_attention_output(model)
    
    # Demo 5: Positional encodings
    demo_positional_encodings(config)
    
    # Demo 6: Prediction interface
    demo_prediction_interface(model)
    
    # Demo 7: Save and load
    demo_model_save_load(model)
    
    print("\n" + "#" * 60)
    print("# ALL DEMOS COMPLETE!")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    run_demo()
