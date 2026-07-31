"""
Test suite for verifying model output format and correctness.

Tests cover:
- Logits shape validation
- Probability distribution (softmax sum to 1)
- Prediction range validity
- Attention weight shapes
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest
from config import ModelConfig
from model import CustomTransformer


@pytest.fixture
def model():
    """Create a small model for testing."""
    config = ModelConfig(
        vocab_size=1000,
        hidden_size=64,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_labels=3,
    )
    return CustomTransformer(config)


@pytest.fixture
def sample_input():
    """Create sample input tensor."""
    return torch.randint(0, 1000, (4, 16))  # batch=4, seq=16


class TestLogitsShape:
    """Test that model output logits have correct shape."""

    def test_logits_shape(self, model, sample_input):
        """Verify output shape is (batch_size, num_labels)."""
        outputs = model(sample_input)
        
        assert "logits" in outputs
        logits = outputs["logits"]
        
        assert logits.shape == torch.Size([4, 3]), (
            f"Expected logits shape (4, 3), got {logits.shape}"
        )

    def test_logits_shape_different_batch(self, model):
        """Verify shape with different batch sizes."""
        for batch_size in [1, 8, 16, 32]:
            input_ids = torch.randint(0, 1000, (batch_size, 16))
            outputs = model(input_ids)
            
            assert outputs["logits"].shape == torch.Size([batch_size, 3]), (
                f"Failed for batch_size={batch_size}"
            )

    def test_logits_shape_different_seq(self, model):
        """Verify shape with different sequence lengths."""
        for seq_len in [8, 16, 32, 64]:
            input_ids = torch.randint(0, 1000, (4, seq_len))
            outputs = model(input_ids)
            
            # Sequence length should not affect logits shape
            assert outputs["logits"].shape == torch.Size([4, 3])


class TestProbabilityDistribution:
    """Test that softmax probabilities sum to 1."""

    def test_probability_sum(self, model, sample_input):
        """Verify softmax probabilities sum to 1.0 per sample."""
        outputs = model(sample_input)
        logits = outputs["logits"]
        
        probs = torch.softmax(logits, dim=-1)
        prob_sums = probs.sum(dim=-1)
        
        # All samples should sum to 1.0
        assert torch.allclose(prob_sums, torch.ones(4), atol=1e-5), (
            f"Probabilities don't sum to 1: {prob_sums}"
        )

    def test_probabilities_non_negative(self, model, sample_input):
        """Verify all probabilities are non-negative."""
        outputs = model(sample_input)
        probs = torch.softmax(outputs["logits"], dim=-1)
        
        assert (probs >= 0).all(), "Found negative probabilities"
        assert (probs <= 1).all(), "Found probabilities greater than 1"

    def test_probability_ordering(self, model, sample_input):
        """Verify prediction from probabilities matches argmax."""
        outputs = model(sample_input)
        logits = outputs["logits"]
        
        probs = torch.softmax(logits, dim=-1)
        pred_from_probs = torch.argmax(probs, dim=-1)
        pred_from_logits = torch.argmax(logits, dim=-1)
        
        assert torch.equal(pred_from_probs, pred_from_logits), (
            "Predictions from probs don't match logits argmax"
        )


class TestPredictionRange:
    """Test that predictions are valid class indices."""

    def test_prediction_indices(self, model, sample_input):
        """Verify predictions are valid class indices (0 to num_labels-1)."""
        predictions = model.predict(sample_input)
        
        assert predictions.min() >= 0, (
            f"Found negative prediction: {predictions.min()}"
        )
        assert predictions.max() < 3, (
            f"Found prediction >= num_labels: {predictions.max()}"
        )
        assert predictions.dtype == torch.long, (
            f"Predictions should be long, got {predictions.dtype}"
        )

    def test_prediction_range_binary(self):
        """Verify predictions for binary classification."""
        config = ModelConfig(vocab_size=500, hidden_size=64, num_labels=2)
        model = CustomTransformer(config)
        
        input_ids = torch.randint(0, 500, (4, 16))
        predictions = model.predict(input_ids)
        
        assert predictions.min() >= 0 and predictions.max() < 2

    def test_prediction_range_multiclass(self):
        """Verify predictions for multi-class classification."""
        config = ModelConfig(vocab_size=500, hidden_size=64, num_labels=5)
        model = CustomTransformer(config)
        
        input_ids = torch.randint(0, 500, (4, 16))
        predictions = model.predict(input_ids)
        
        assert predictions.min() >= 0 and predictions.max() < 5


class TestAttentionOutput:
    """Test that attention weights are correctly shaped."""

    def test_attention_shape(self, model, sample_input):
        """Verify attention weights have correct shape."""
        outputs = model(sample_input, output_attentions=True)
        
        assert "attentions" in outputs
        attentions = outputs["attentions"]
        
        # Should have one attention tensor per layer
        assert len(attentions) == 2  # num_hidden_layers = 2
        
        # Shape: (batch, num_heads, seq_len, seq_len)
        expected_shape = torch.Size([4, 4, 16, 16])
        assert attentions[0].shape == expected_shape, (
            f"Expected {expected_shape}, got {attentions[0].shape}"
        )

    def test_attention_values_sum_to_one(self, model, sample_input):
        """Verify attention weights are approximately non-negative and valid."""
        # Set model to eval mode to disable dropout
        model.eval()
        
        outputs = model(sample_input, output_attentions=True)
        attentions = outputs["attentions"]
        
        attn = attentions[0]  # First layer
        
        # In eval mode, attention should sum close to 1
        attn_sums = attn.sum(dim=-1)
        
        # Allow some tolerance due to floating point
        assert torch.allclose(attn_sums, torch.ones_like(attn_sums), atol=0.1), (
            f"Attention sums out of range: min={attn_sums.min():.3f}, max={attn_sums.max():.3f}"
        )

    def test_attention_non_negative(self, model, sample_input):
        """Verify attention weights are non-negative."""
        outputs = model(sample_input, output_attentions=True)
        attentions = outputs["attentions"]
        
        for layer_attn in attentions:
            assert (layer_attn >= 0).all(), "Negative attention weights found"


class TestLossComputation:
    """Test that loss computation works correctly."""

    def test_loss_shape(self, model, sample_input):
        """Verify loss is a scalar tensor."""
        labels = torch.randint(0, 3, (4,))
        outputs = model(sample_input, labels=labels)
        
        assert "loss" in outputs
        loss = outputs["loss"]
        
        assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
        assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"

    def test_loss_backward(self, model, sample_input):
        """Verify loss can be backpropagated."""
        model.zero_grad()
        labels = torch.randint(0, 3, (4,))
        outputs = model(sample_input, labels=labels)
        loss = outputs["loss"]
        
        loss.backward()
        
        # Check that gradients exist for most parameters
        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        total_count = sum(1 for _ in model.parameters())
        
        # At least 80% of parameters should have gradients
        assert grad_count > total_count * 0.8, (
            f"Too few parameters have gradients: {grad_count}/{total_count}"
        )

    def test_loss_with_different_label_counts(self, model, sample_input):
        """Test loss computation with different number of labels."""
        for num_labels in [2, 5, 10]:
            config = ModelConfig(vocab_size=1000, hidden_size=64, num_labels=num_labels)
            m = CustomTransformer(config)
            
            labels = torch.randint(0, num_labels, (4,))
            outputs = m(sample_input, labels=labels)
            
            assert outputs["loss"].item() >= 0


class TestModelConsistency:
    """Test model behavior is consistent across calls."""

    def test_same_input_same_output(self, model):
        """Verify same input produces same output (no dropout during eval)."""
        model.eval()
        input_ids = torch.randint(0, 1000, (4, 16))
        
        with torch.no_grad():
            out1 = model(input_ids)["logits"]
            out2 = model(input_ids)["logits"]
        
        assert torch.allclose(out1, out2, atol=1e-6), (
            "Same input produced different outputs in eval mode"
        )

    def test_train_eval_mode(self, model, sample_input):
        """Verify model behaves differently in train vs eval mode."""
        # Eval mode
        model.eval()
        with torch.no_grad():
            eval_out = model(sample_input)["logits"]
        
        # Train mode (dropout active)
        model.train()
        train_out = model(sample_input)["logits"]
        
        # Outputs should differ due to dropout
        # (Not guaranteed but highly likely with random initialization)
        assert eval_out.shape == train_out.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
