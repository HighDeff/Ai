"""
Main Training Script for Custom Transformer Model.

This script demonstrates:
- Model creation with custom configuration
- Data generation and preprocessing
- Training loop
- Evaluation
- Inference examples

Usage:
    python train.py [--config CONFIG_PATH] [--epochs EPOCHS] [--batch_size BS]
"""

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import ModelConfig, TrainingConfig, DataConfig
from model import CustomTransformer
from dataset import (
    Vocabulary,
    TextClassificationDataset,
    DataCollator,
    generate_sample_data,
    split_data,
    create_simple_dataloader,
)
from trainer import (
    Trainer,
    EarlyStopping,
    CosineAnnealingWarmupScheduler,
    create_optimizer,
    TrainingMetrics,
    EvaluationMetrics,
    format_metrics,
)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_model_summary(model: CustomTransformer):
    """Print model architecture summary."""
    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE")
    print("=" * 60)
    
    total_params = model.get_num_params()
    trainable_params = model.get_num_params(trainable_only=True)
    
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
    
    print("\nComponents:")
    print(f"  - Embeddings: {model.embeddings.__class__.__name__}")
    print(f"  - Encoder Layers: {model.config.num_hidden_layers}")
    print(f"  - Attention Heads: {model.config.num_attention_heads}")
    print(f"  - Hidden Size: {model.config.hidden_size}")
    print(f"  - Vocabulary Size: {model.config.vocab_size}")
    
    print("\n" + "=" * 60)


def logging_callback(metrics: TrainingMetrics):
    """Callback function for logging training metrics."""
    print(
        f"Step {metrics.step} | "
        f"Loss: {metrics.loss:.4f} | "
        f"Acc: {metrics.accuracy:.4f} | "
        f"LR: {metrics.learning_rate:.2e}"
    )


def train(args) -> CustomTransformer:
    """
    Main training function.
    
    Args:
        args: Command line arguments
    
    Returns:
        Trained model
    """
    # Set seed
    set_seed(args.seed)
    
    print("\n" + "=" * 60)
    print("CUSTOM TRANSFORMER MODEL TRAINING")
    print("=" * 60)
    
    # Configuration
    model_config = ModelConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_heads,
        num_hidden_layers=args.num_layers,
        intermediate_size=args.intermediate_size,
        num_labels=args.num_labels,
        hidden_dropout_prob=args.dropout,
        attention_dropout_prob=args.dropout,
        pooling_type=args.pooling,
    )
    
    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
    )
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Create model
    print("\nInitializing model...")
    model = CustomTransformer(model_config)
    model.to(device)
    print_model_summary(model)
    
    # Generate sample data
    print("\nGenerating sample data...")
    texts, labels = generate_sample_data(
        num_samples=args.num_samples,
        num_classes=args.num_labels,
        avg_text_length=args.avg_text_length,
        seed=args.seed,
    )
    
    print(f"Total samples: {len(texts)}")
    print(f"Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    # Split data
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = split_data(
        texts, labels,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    
    print(f"\nData splits:")
    print(f"  Train: {len(train_texts)} samples")
    print(f"  Val: {len(val_texts)} samples")
    print(f"  Test: {len(test_texts)} samples")
    
    # Build vocabulary
    print("\nBuilding vocabulary...")
    vocab = Vocabulary(min_freq=2, max_vocab_size=args.vocab_size)
    vocab.build_vocab(train_texts)
    print(f"Vocabulary size: {vocab.vocab_size}")
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader = create_simple_dataloader(
        texts=train_texts,
        labels=train_labels,
        vocab=vocab,
        batch_size=args.batch_size,
        max_length=args.max_seq_length,
        shuffle=True,
    )
    
    val_loader = create_simple_dataloader(
        texts=val_texts,
        labels=val_labels,
        vocab=vocab,
        batch_size=args.batch_size,
        max_length=args.max_seq_length,
        shuffle=False,
    )
    
    test_loader = create_simple_dataloader(
        texts=test_texts,
        labels=test_labels,
        vocab=vocab,
        batch_size=args.batch_size,
        max_length=args.max_seq_length,
        shuffle=False,
    )
    
    # Create optimizer
    print("\nSetting up optimizer...")
    optimizer = create_optimizer(
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    # Create scheduler
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    scheduler = CosineAnnealingWarmupScheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr=args.min_lr,
    )
    
    # Create early stopping
    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta,
        mode="min",
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        gradient_accumulation_steps=args.gradient_accumulation,
        max_grad_norm=args.max_grad_norm,
        scheduler=scheduler,
        early_stopping=early_stopping,
    )
    
    # Training
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    history = trainer.train(
        train_loader=train_loader,
        eval_loader=val_loader,
        num_epochs=args.epochs,
        logging_fn=logging_callback if args.verbose else None,
        logging_steps=args.logging_steps,
        save_path=args.save_path,
    )
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)
    
    test_metrics = trainer.evaluate(test_loader)
    print(f"\nTest Metrics: {format_metrics(test_metrics)}")
    
    # Save final model
    if args.save_path:
        final_path = args.save_path.replace(".pt", "_final.pt")
        torch.save({
            "config": model_config,
            "state_dict": model.state_dict(),
            "vocab_size": vocab.vocab_size,
            "test_metrics": {
                "loss": test_metrics.loss,
                "accuracy": test_metrics.accuracy,
                "f1": test_metrics.f1,
            },
        }, final_path)
        print(f"\nFinal model saved to: {final_path}")
    
    return model, vocab, test_metrics


def run_inference(
    model: CustomTransformer,
    vocab: Vocabulary,
    device: torch.device,
    sample_texts: List[str],
) -> None:
    """
    Run inference on sample texts.
    
    Args:
        model: Trained model
        vocab: Vocabulary
        device: Device to run on
        sample_texts: List of sample texts
    """
    print("\n" + "=" * 60)
    print("SAMPLE INFERENCE")
    print("=" * 60)
    
    model.eval()
    
    for text in sample_texts:
        print(f"\nInput: {text}")
        
        # Encode
        token_ids = vocab.encode(text, max_length=128)
        input_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_ids=input_tensor)
            probs = torch.softmax(outputs["logits"], dim=-1)
            pred_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, pred_class].item()
        
        class_names = ["Class 0 (Science)", "Class 1 (Sports)", "Class 2 (Business)", "Class 3 (Entertainment)"]
        print(f"Prediction: {class_names[min(pred_class, 3)]}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Probabilities: {probs[0].cpu().numpy()}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train Custom Transformer Model for Text Classification"
    )
    
    # Model arguments
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of encoder layers")
    parser.add_argument("--intermediate_size", type=int, default=1024, help="FFN intermediate size")
    parser.add_argument("--num_labels", type=int, default=2, help="Number of classification labels")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "cls"], help="Pooling type")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--min_lr", type=float, default=1e-7, help="Minimum learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--gradient_accumulation", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--min_delta", type=float, default=1e-4, help="Early stopping min delta")
    
    # Data arguments
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--avg_text_length", type=int, default=15, help="Average text length")
    parser.add_argument("--max_seq_length", type=int, default=64, help="Maximum sequence length")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training split ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test split ratio")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--save_path", type=str, default="model_checkpoint.pt", help="Save path")
    parser.add_argument("--logging_steps", type=int, default=50, help="Logging interval")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Validate ratios
    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6, (
        "Split ratios must sum to 1.0"
    )
    
    # Train model
    model, vocab, metrics = train(args)
    
    # Run sample inference
    sample_texts = [
        "science research experiment data study analysis",
        "sports game team player match championship score",
        "business market company investment stock economy",
        "entertainment movie music celebrity film show",
    ]
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_inference(model, vocab, device, sample_texts)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
