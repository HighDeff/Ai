# Custom PyTorch Transformer Model Specification

## Project Overview
- **Project Name**: CustomTransformer - A Transformer-based Text Classifier
- **Type**: Deep Learning Model (PyTorch)
- **Core Functionality**: Custom transformer architecture with multi-head attention, positional encoding, and classification head for text classification tasks
- **Target Users**: ML practitioners, researchers, developers learning transformer architectures

## Model Architecture

### Core Components
1. **Embedding Layer**: Token embeddings + learnable positional encodings
2. **Transformer Encoder Block**:
   - Multi-head self-attention with scaled dot-product
   - Feed-forward network (FFN) with GELU activation
   - Residual connections & layer normalization
3. **Classification Head**:
   - Global average pooling
   - Dropout for regularization
   - Dense layer with softmax for multi-class output

### Architecture Parameters
- `vocab_size`: 30522 (BERT tokenizer vocab)
- `hidden_size`: 256 (embedding dimension)
- `num_attention_heads`: 8
- `num_hidden_layers`: 4
- `intermediate_size`: 512 (FFN hidden size)
- `hidden_dropout_prob`: 0.1
- `attention_dropout_prob`: 0.1
- `num_labels`: 2 (binary classification, configurable)

## Features

### Custom Components
1. **Scaled Dot-Product Attention**: Implementation of multi-head attention from scratch
2. **Positional Encoding**: Sinusoidal and learnable positional encodings
3. **Transformer Encoder Layer**: Complete encoder block with pre-norm architecture
4. **Custom Model Class**: Full nn.Module subclass with training/inference modes

### Utility Functions
- Weight initialization (Xavier/Glorot)
- Gradient clipping
- Model summary printer
- Checkpoint saving/loading
- Forward pass with attention weights extraction

### Training Pipeline
- Data loading with custom Dataset class
- Training loop with gradient accumulation
- Evaluation metrics (accuracy, precision, recall, F1)
- Learning rate scheduling (cosine annealing)
- Early stopping

## Dataset System

### Supported Datasets
1. **IMDB Sentiment**: Binary classification (positive/negative)
2. **Topic Classification**: Multi-class with Q&A format (Science, Sports, Business, Technology)
3. **Synthetic**: Generated sample data for testing

### Dataset Structure
```
datasets/
├── __init__.py           # Package exports
├── imdb_dataset.py       # IMDB sentiment loader
├── topic_dataset.py      # Topic classification loader
└── expected_results.py   # Expected results schema
```

### Expected Results Schema
- `task_type`: classification, qa, regression
- `dataset_name`: Name of the dataset
- `num_classes`: Number of classification classes
- `class_names`: Names of each class
- `sample_inputs`: List of sample texts
- `expected_labels`: Ground truth labels
- `expected_probabilities`: Optional probability distributions

## Testing System

### Test Files
```
tests/
├── __init__.py              # Package init
├── test_model_output.py     # Model output validation
└── test_dataset.py           # Dataset validation
```

### Verification Tests
| Test | Expected Result |
|------|-----------------|
| `test_logits_shape` | Output shape is `(batch_size, num_labels)` |
| `test_probability_sum` | Softmax output sums to 1.0 per sample |
| `test_prediction_range` | Predictions are integers in `[0, num_labels-1]` |
| `test_attention_output` | Attention weights shape correct |
| `test_vocab_alignment` | Vocabulary maps correctly |
| `test_label_range` | Labels within valid range |
| `test_train_test_split` | Data splits sum correctly |
| `test_batch_output` | Model output matches num_labels |

### Evaluation Reporter
- `ResultsReporter` class tracks predictions vs expected
- Generates accuracy, per-class metrics (precision, recall, F1)
- Logs misclassified examples for analysis
- Exports to JSON for inspection

## File Structure
```
pytorch_model/
├── model.py              # Main model architecture
├── attention.py          # Custom attention mechanisms
├── embeddings.py         # Token & positional embeddings
├── trainer.py            # Training utilities
├── dataset.py            # Data loading utilities
├── config.py             # Model configuration
├── train.py              # Main training script
├── demo.py               # Demo/inference script
├── evaluate_results.py   # Results reporter
├── datasets/             # Dataset loaders
│   ├── __init__.py
│   ├── imdb_dataset.py
│   ├── topic_dataset.py
│   └── expected_results.py
└── tests/                # Test suite
    ├── __init__.py
    ├── test_model_output.py
    └── test_dataset.py
```

## Dependencies
- torch >= 2.0
- numpy
- tqdm (progress bars)
- pytest (for testing)
