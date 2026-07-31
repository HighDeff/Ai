"""
Training utilities for the Custom Transformer Model.

This module provides:
- Training loop with gradient accumulation
- Evaluation metrics
- Learning rate schedulers
- Early stopping
- Checkpoint management
"""

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    epoch: int = 0
    step: int = 0
    loss: float = 0.0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    learning_rate: float = 0.0
    elapsed_time: float = 0.0


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    loss: float = 0.0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    num_samples: int = 0


class EarlyStopping:
    """
    Early stopping handler to stop training when validation loss stops improving.
    
    Args:
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement
        mode: 'min' or 'max' for comparing metric
        save_path: Path to save best model
    """
    
    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 1e-4,
        mode: str = "min",
        save_path: Optional[str] = None,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.save_path = save_path
        
        self.best_value: Optional[float] = None
        self.counter = 0
        self.should_stop = False
        
        if mode == "min":
            self.is_better = lambda current, best: current < (best - min_delta)
        else:
            self.is_better = lambda current, best: current > (best + min_delta)
    
    def __call__(
        self, 
        current_value: float, 
        model: Optional[nn.Module] = None
    ) -> bool:
        """
        Check if training should stop.
        
        Args:
            current_value: Current metric value
            model: Optional model to save
        
        Returns:
            True if training should stop
        """
        if self.best_value is None:
            self.best_value = current_value
            self._save_if_needed(model, current_value)
            return False
        
        if self.is_better(current_value, self.best_value):
            self.best_value = current_value
            self.counter = 0
            self._save_if_needed(model, current_value)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        
        return False
    
    def _save_if_needed(self, model: Optional[nn.Module], value: float):
        """Save model if path provided and value is best."""
        if model is not None and self.save_path:
            torch.save(model.state_dict(), self.save_path)


class CosineAnnealingWarmupScheduler:
    """
    Learning rate scheduler with cosine annealing and warmup.
    
    Schedule:
        - Linear warmup for warmup_steps
        - Cosine annealing to min_lr
    
    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        min_lr: Minimum learning rate
        max_lr: Maximum learning rate (will be set from optimizer)
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-7,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.current_step = 0
    
    def step(self):
        """Update learning rate."""
        self.current_step += 1
        new_lrs = self.get_last_lr()
        for param_group, lr in zip(self.optimizer.param_groups, new_lrs):
            param_group["lr"] = lr
    
    def get_last_lr(self) -> List[float]:
        """Get current learning rates."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return [
                base_lr * self.current_step / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            return [
                self.min_lr + (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress)) / 2
                for base_lr in self.base_lrs
            ]


def compute_metrics(
    predictions: Tensor,
    labels: Tensor,
    loss: Optional[float] = None,
) -> EvaluationMetrics:
    """
    Compute classification metrics.
    
    Args:
        predictions: Predicted class indices
        labels: Ground truth labels
        loss: Optional loss value to include
    
    Returns:
        EvaluationMetrics object
    """
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    
    # Accuracy
    accuracy = (predictions == labels).mean()
    
    # Per-class metrics
    num_classes = max(predictions.max(), labels.max()) + 1
    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    
    for c in range(num_classes):
        pred_c = predictions == c
        label_c = labels == c
        
        tp = (pred_c & label_c).sum()
        fp = (pred_c & ~label_c).sum()
        fn = (~pred_c & label_c).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1
    
    # Macro average
    n = max(num_classes, 1)
    
    metrics = EvaluationMetrics(
        loss=loss if loss is not None else 0.0,
        accuracy=accuracy,
        precision=precision_sum / n,
        recall=recall_sum / n,
        f1=f1_sum / n,
        num_samples=len(labels),
    )
    
    return metrics


def format_metrics(metrics: EvaluationMetrics) -> str:
    """Format metrics as a readable string."""
    return (
        f"Loss: {metrics.loss:.4f} | "
        f"Acc: {metrics.accuracy:.4f} | "
        f"Prec: {metrics.precision:.4f} | "
        f"Rec: {metrics.recall:.4f} | "
        f"F1: {metrics.f1:.4f}"
    )


class Trainer:
    """
    Training loop manager for the transformer model.
    
    Handles:
        - Training loop with gradient accumulation
        - Evaluation
        - Checkpointing
        - Logging
        - Early stopping
    
    Args:
        model: The model to train
        optimizer: PyTorch optimizer
        device: Device to train on ('cuda' or 'cpu')
        gradient_accumulation_steps: Number of steps to accumulate gradients
        max_grad_norm: Maximum gradient norm for clipping
        scheduler: Optional learning rate scheduler
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: Union[str, torch.device] = "cuda",
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        scheduler: Optional[Any] = None,
        early_stopping: Optional[EarlyStopping] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = torch.device(device) if isinstance(device, str) else device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float("inf")
        
        # Move model to device
        self.model.to(self.device)
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        logging_fn: Optional[Callable[[TrainingMetrics], None]] = None,
        logging_steps: int = 100,
    ) -> TrainingMetrics:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            logging_fn: Optional function to log metrics
            logging_steps: Log every n steps
        
        Returns:
            Final TrainingMetrics for the epoch
        """
        self.model.train()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        epoch_metrics = TrainingMetrics(epoch=self.epoch)
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch + 1}")
        
        for step, batch in enumerate(pbar):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            loss = outputs["loss"]
            logits = outputs["logits"]
            
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Get predictions
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.append(predictions.detach().cpu())
            all_labels.append(labels.detach().cpu())
            
            # Gradient accumulation
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.max_grad_norm
                )
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Scheduler step
                if self.scheduler is not None:
                    self.scheduler.step()
                
                self.global_step += 1
            
            # Update metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            
            # Logging
            if logging_fn is not None and (step + 1) % logging_steps == 0:
                epoch_metrics.loss = total_loss / (step + 1)
                epoch_metrics.step = self.global_step
                epoch_metrics.learning_rate = self.optimizer.param_groups[0]["lr"]
                logging_fn(epoch_metrics)
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })
        
        # Compute final epoch metrics
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        metrics = compute_metrics(all_predictions, all_labels, total_loss / len(train_loader))
        
        return TrainingMetrics(
            epoch=self.epoch,
            step=self.global_step,
            loss=metrics.loss,
            accuracy=metrics.accuracy,
            precision=metrics.precision,
            recall=metrics.recall,
            f1=metrics.f1,
            learning_rate=self.optimizer.param_groups[0]["lr"],
        )
    
    @torch.no_grad()
    def evaluate(
        self,
        eval_loader: DataLoader,
    ) -> EvaluationMetrics:
        """
        Evaluate the model.
        
        Args:
            eval_loader: Evaluation data loader
        
        Returns:
            EvaluationMetrics
        """
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        for batch in tqdm(eval_loader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            total_loss += outputs["loss"].item()
            
            # Get predictions
            predictions = torch.argmax(outputs["logits"], dim=-1)
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
        
        # Concatenate all predictions and labels
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        
        return compute_metrics(
            all_predictions, 
            all_labels, 
            total_loss / len(eval_loader)
        )
    
    def train(
        self,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        num_epochs: int = 10,
        logging_fn: Optional[Callable[[TrainingMetrics], None]] = None,
        logging_steps: int = 100,
        save_path: Optional[str] = None,
    ) -> Dict[str, List[EvaluationMetrics]]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            eval_loader: Optional evaluation data loader
            num_epochs: Number of epochs to train
            logging_fn: Optional function to log training metrics
            logging_steps: Log every n steps
            save_path: Optional path to save best model
        
        Returns:
            Dictionary with 'train' and 'eval' metric histories
        """
        history = {"train": [], "eval": []}
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(
                train_loader=train_loader,
                logging_fn=logging_fn,
                logging_steps=logging_steps,
            )
            history["train"].append(train_metrics)
            
            # Evaluate
            if eval_loader is not None:
                eval_metrics = self.evaluate(eval_loader)
                history["eval"].append(eval_metrics)
                
                # Print evaluation results
                print(f"\nEpoch {epoch + 1} - Eval: {format_metrics(eval_metrics)}")
                
                # Check for early stopping
                if self.early_stopping is not None:
                    should_stop = self.early_stopping(
                        eval_metrics.loss, 
                        self.model
                    )
                    if should_stop:
                        print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                        break
                
                # Save best model
                if save_path and eval_metrics.loss < self.best_metric:
                    self.best_metric = eval_metrics.loss
                    self.save_checkpoint(save_path, eval_metrics)
                    print(f"Saved best model to {save_path}")
        
        return history
    
    def save_checkpoint(
        self, 
        path: str, 
        metrics: Optional[EvaluationMetrics] = None,
    ):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        
        if metrics is not None:
            checkpoint["metrics"] = {
                "loss": metrics.loss,
                "accuracy": metrics.accuracy,
                "f1": metrics.f1,
            }
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        
        print(f"Loaded checkpoint from epoch {self.epoch}")


def create_optimizer(
    model: nn.Module,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    adam_epsilon: float = 1e-8,
    beta1: float = 0.9,
    beta2: float = 0.999,
) -> torch.optim.Optimizer:
    """
    Create AdamW optimizer with weight decay.
    
    Args:
        model: Model to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay coefficient
        adam_epsilon: Epsilon for Adam
        beta1: Beta1 for Adam
        beta2: Beta2 for Adam
    
    Returns:
        AdamW optimizer
    """
    # Separate parameters with and without weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Don't apply weight decay to bias and LayerNorm weights
        if "bias" in name or "layer_norm" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    optimizer_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    
    return torch.optim.AdamW(
        optimizer_groups,
        lr=learning_rate,
        eps=adam_epsilon,
        betas=(beta1, beta2),
    )
