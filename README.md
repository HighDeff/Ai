# AI Companion

A comprehensive AI application featuring a custom chat interface and a PyTorch transformer model.

## Apps

### 1. AI Chat Application (`index.html`)
A modern conversational AI interface with:
- **Three Modes**: Chat, Q&A, and Prediction
- **Modern dark theme** with gradient accents
- **Demo mode** (works without API key)
- **API integration** with OpenAI-compatible endpoints
- **Conversation history** with local storage
- **Responsive design** for all devices

**Usage**: Open `index.html` in a browser.

### 2. Custom PyTorch Transformer Model (`pytorch_model/`)
A complete transformer-based text classifier built from scratch.

**Features**:
- Custom multi-head attention mechanism
- Sinusoidal and learnable positional encodings
- Transformer encoder layers with pre-norm
- Text classification (binary and multi-class)
- Training pipeline with metrics

**Run Demo**:
```bash
cd pytorch_model
python demo.py
```

**Train Model**:
```bash
python train.py --epochs 5 --batch_size 32
```

**Run Tests**:
```bash
pytest tests/ -v
```

## Structure

```
Ai/
├── index.html              # AI Chat web application
├── pytorch_model/         # Custom PyTorch transformer
│   ├── model.py           # Main transformer model
│   ├── attention.py       # Custom attention mechanisms
│   ├── embeddings.py      # Token & positional embeddings
│   ├── trainer.py         # Training utilities
│   ├── dataset.py         # Data loading
│   ├── config.py          # Model configuration
│   ├── train.py           # Training script
│   ├── demo.py            # Demo script
│   ├── evaluate_results.py # Results reporter
│   ├── datasets/          # Dataset loaders
│   │   ├── imdb_dataset.py
│   │   ├── topic_dataset.py
│   │   └── expected_results.py
│   └── tests/             # Test suite
│       ├── test_model_output.py
│       └── test_dataset.py
├── LICENSE
└── README.md
```
