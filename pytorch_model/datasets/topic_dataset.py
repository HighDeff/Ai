"""
Topic Classification Dataset Loader with Q&A pairs.

Supports:
- Custom topic classification with explicit question-context-answer format
- Multi-class classification (Science, Sports, Business, Technology, etc.)
- Train/test splitting
- Integration with existing Dataset class

Expected format:
- Question: The input question/query
- Context: Optional supporting context
- Answer: The expected answer
- Label: Class index (0 to num_classes-1)
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

import torch
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset import Vocabulary, DataCollator


@dataclass
class TopicSample:
    """Single topic classification sample."""
    question: str
    context: Optional[str] = None
    answer: Optional[str] = None
    label: int = 0
    topic_name: str = ""


@dataclass
class TopicExpectedResults:
    """Expected results for topic classification dataset."""
    task_type: str = "classification"
    dataset_name: str = "TopicClassification"
    num_classes: int = 4
    class_names: List[str] = None
    sample_inputs: List[str] = None
    expected_labels: List[int] = None
    
    def __post_init__(self):
        if self.class_names is None:
            self.class_names = ["Science", "Sports", "Business", "Technology"]
        if self.sample_inputs is None:
            self.sample_inputs = []
        if self.expected_labels is None:
            self.expected_labels = []


class TopicDataset(Dataset):
    """
    Topic Classification Dataset with question-context-answer format.
    
    Args:
        samples: List of TopicSample objects
        vocab: Vocabulary for tokenization
        max_length: Maximum sequence length
        include_context: Whether to include context in input
    """
    
    def __init__(
        self,
        samples: List[TopicSample],
        vocab: Vocabulary,
        max_length: int = 128,
        include_context: bool = False,
    ):
        self.samples = samples
        self.vocab = vocab
        self.max_length = max_length
        self.include_context = include_context
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample."""
        sample = self.samples[idx]
        
        # Build input text
        if self.include_context and sample.context:
            input_text = f"{sample.question} {sample.context}"
        else:
            input_text = sample.question
        
        token_ids = self.vocab.encode(input_text, max_length=self.max_length)
        label = sample.label
        
        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )
    
    def get_raw_sample(self, idx: int) -> TopicSample:
        """Get raw sample without tokenization."""
        return self.samples[idx]


class TopicDataLoader:
    """
    Data loader for topic classification with Q&A format.
    
    Example:
        >>> loader = TopicDataLoader()
        >>> train_loader, test_loader = loader.load_topic_data()
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        max_length: int = 128,
        batch_size: int = 32,
        train_ratio: float = 0.8,
        seed: int = 42,
    ):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.seed = seed
        
        self.vocab = None
        self.train_loader = None
        self.test_loader = None
        self.samples = []
    
    def _get_sample_data(self) -> List[TopicSample]:
        """Generate sample topic classification data."""
        samples = [
            # Science (label=0)
            TopicSample(
                question="What causes global warming?",
                context="The greenhouse effect and carbon emissions",
                answer="The greenhouse effect caused by greenhouse gases",
                label=0,
                topic_name="Science"
            ),
            TopicSample(
                question="Explain how vaccines work",
                context="The immune system response to antigens",
                answer="They trigger an immune response without causing disease",
                label=0,
                topic_name="Science"
            ),
            TopicSample(
                question="What is the theory of relativity?",
                context="Einstein's physics theory",
                answer="Space and time are relative to the observer's motion",
                label=0,
                topic_name="Science"
            ),
            TopicSample(
                question="How do computers process information?",
                context="Binary code and logic gates",
                answer="Using binary code and logical operations",
                label=0,
                topic_name="Science"
            ),
            TopicSample(
                question="What is DNA and its function?",
                context="Genetic material in cells",
                answer="DNA carries genetic instructions for development",
                label=0,
                topic_name="Science"
            ),
            
            # Sports (label=1)
            TopicSample(
                question="Who won the World Cup in 2022?",
                context="FIFA World Cup tournament",
                answer="Argentina won the 2022 FIFA World Cup",
                label=1,
                topic_name="Sports"
            ),
            TopicSample(
                question="What is the offside rule in soccer?",
                context="Football rules and regulations",
                answer="Being closer to the opponent's goal than the ball and second-last defender",
                label=1,
                topic_name="Sports"
            ),
            TopicSample(
                question="How many players are on a basketball team?",
                context="Basketball game rules",
                answer="Five players on the court per team",
                label=1,
                topic_name="Sports"
            ),
            TopicSample(
                question="What sport uses a shuttlecock?",
                context="Racket sports",
                answer="Badminton uses a shuttlecock",
                label=1,
                topic_name="Sports"
            ),
            TopicSample(
                question="Who holds the record for most Olympic gold medals?",
                context="Olympic history and records",
                answer="Michael Phelps holds the record for most Olympic gold medals",
                label=1,
                topic_name="Sports"
            ),
            
            # Business (label=2)
            TopicSample(
                question="What is a stock market index?",
                context="Financial markets and investments",
                answer="A measurement of a section of the stock market",
                label=2,
                topic_name="Business"
            ),
            TopicSample(
                question="How does compound interest work?",
                context="Banking and finance",
                answer="Interest calculated on initial principal and accumulated interest",
                label=2,
                topic_name="Business"
            ),
            TopicSample(
                question="What is inflation?",
                context="Economic concepts",
                answer="The rate at which prices for goods and services rise",
                label=2,
                topic_name="Business"
            ),
            TopicSample(
                question="What is GDP?",
                context="Economic indicators",
                answer="Gross Domestic Product - total value of goods and services produced",
                label=2,
                topic_name="Business"
            ),
            TopicSample(
                question="Explain what a bond is in finance",
                context="Investment instruments",
                answer="A debt security where an investor loans money to an entity",
                label=2,
                topic_name="Business"
            ),
            
            # Technology (label=3)
            TopicSample(
                question="What is artificial intelligence?",
                context="Computer science and ML",
                answer="Computer systems that can perform tasks requiring human intelligence",
                label=3,
                topic_name="Technology"
            ),
            TopicSample(
                question="How does blockchain work?",
                context="Cryptocurrency and distributed ledgers",
                answer="A decentralized ledger of transactions across many computers",
                label=3,
                topic_name="Technology"
            ),
            TopicSample(
                question="What is cloud computing?",
                context="Internet-based computing services",
                answer="Delivery of computing services over the internet",
                label=3,
                topic_name="Technology"
            ),
            TopicSample(
                question="Explain machine learning",
                context="AI subset and algorithms",
                answer="Algorithms that learn patterns from data without explicit programming",
                label=3,
                topic_name="Technology"
            ),
            TopicSample(
                question="What is the Internet of Things?",
                context="Connected devices and networks",
                answer="Physical devices with sensors and software that connect and exchange data",
                label=3,
                topic_name="Technology"
            ),
        ]
        
        # Generate more samples by creating variations
        expanded_samples = []
        
        science_templates = [
            "What is {}?",
            "How does {} work?",
            "Explain the concept of {}",
        ]
        
        science_topics = [
            "photosynthesis", "evolution", "quantum mechanics", "thermodynamics",
            "neuroscience", "genetics", "astronomy", "chemistry"
        ]
        
        sports_topics = [
            "marathon running", "tennis scoring", "golf handicaps", "swimming strokes",
            "cycling races", "boxing scoring", "ski jumping", "archery"
        ]
        
        business_topics = [
            "supply and demand", "market capitalization", "dividends", "bull markets",
            "IPO", "venture capital", "brand equity", "market segmentation"
        ]
        
        tech_topics = [
            "cybersecurity", "5G networks", "virtual reality", "edge computing",
            "quantum computing", "neural networks", "natural language processing", "computer vision"
        ]
        
        for topic in science_topics:
            for template in science_templates:
                q = template.format(topic)
                expanded_samples.append(TopicSample(
                    question=q,
                    context="Scientific research and studies",
                    answer=f"Related to {topic}",
                    label=0,
                    topic_name="Science"
                ))
        
        for topic in sports_topics:
            for template in science_templates:
                q = template.format(topic)
                expanded_samples.append(TopicSample(
                    question=q,
                    context="Sports and athletics",
                    answer=f"Related to {topic}",
                    label=1,
                    topic_name="Sports"
                ))
        
        for topic in business_topics:
            for template in science_templates:
                q = template.format(topic)
                expanded_samples.append(TopicSample(
                    question=q,
                    context="Business and economics",
                    answer=f"Related to {topic}",
                    label=2,
                    topic_name="Business"
                ))
        
        for topic in tech_topics:
            for template in science_templates:
                q = template.format(topic)
                expanded_samples.append(TopicSample(
                    question=q,
                    context="Technology and computing",
                    answer=f"Related to {topic}",
                    label=3,
                    topic_name="Technology"
                ))
        
        return samples + expanded_samples
    
    def _try_load_from_json(self, path: str) -> List[TopicSample]:
        """Load topic data from JSON file."""
        if not Path(path).exists():
            return []
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        samples = []
        for item in data:
            samples.append(TopicSample(
                question=item.get('question', ''),
                context=item.get('context'),
                answer=item.get('answer'),
                label=item.get('label', 0),
                topic_name=item.get('topic_name', '')
            ))
        
        return samples
    
    def load_topic_data(
        self,
        json_path: Optional[str] = None,
        include_context: bool = True,
    ) -> Tuple[DataLoader, DataLoader, Vocabulary, List[TopicSample]]:
        """
        Load topic classification data and create data loaders.
        
        Args:
            json_path: Path to JSON file (if None, uses sample data)
            include_context: Whether to include context in input
        
        Returns:
            Tuple of (train_loader, test_loader, vocab, all_samples)
        """
        random.seed(self.seed)
        
        # Load data
        if json_path:
            samples = self._try_load_from_json(json_path)
            if not samples:
                print(f"JSON not found at {json_path}, using sample data")
                samples = self._get_sample_data()
        else:
            samples = self._get_sample_data()
        
        self.samples = samples
        print(f"Loaded {len(samples)} topic classification samples")
        
        # Count labels
        label_counts = {}
        for s in samples:
            label_counts[s.label] = label_counts.get(s.label, 0) + 1
        print(f"Class distribution: {label_counts}")
        
        # Build vocabulary
        texts = [s.question for s in samples]
        if include_context:
            texts += [s.context for s in samples if s.context]
        
        self.vocab = Vocabulary(min_freq=2, max_vocab_size=self.vocab_size)
        self.vocab.build_vocab(texts)
        print(f"Vocabulary size: {self.vocab.vocab_size}")
        
        # Split data
        random.shuffle(samples)
        split_idx = int(len(samples) * self.train_ratio)
        train_samples = samples[:split_idx]
        test_samples = samples[split_idx:]
        
        # Create datasets
        train_dataset = TopicDataset(
            samples=train_samples,
            vocab=self.vocab,
            max_length=self.max_length,
            include_context=include_context,
        )
        
        test_dataset = TopicDataset(
            samples=test_samples,
            vocab=self.vocab,
            max_length=self.max_length,
            include_context=include_context,
        )
        
        # Create data loaders
        collator = DataCollator(vocab=self.vocab)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collator,
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collator,
        )
        
        return self.train_loader, self.test_loader, self.vocab, samples
    
    def get_expected_results(self) -> TopicExpectedResults:
        """Get expected results schema for topic classification."""
        # Get unique classes
        classes = set(s.topic_name for s in self.samples)
        class_names = sorted(list(classes)) if classes else ["Science", "Sports", "Business", "Technology"]
        
        # Get sample inputs with expected labels
        sample_inputs = []
        expected_labels = []
        for s in self.samples[:10]:
            sample_inputs.append(s.question)
            expected_labels.append(s.label)
        
        return TopicExpectedResults(
            task_type="classification",
            dataset_name="TopicClassification",
            num_classes=len(class_names),
            class_names=class_names,
            sample_inputs=sample_inputs,
            expected_labels=expected_labels,
        )


def get_topic_expected_results() -> TopicExpectedResults:
    """Get default expected results for topic classification."""
    return TopicExpectedResults()


# Sample usage
if __name__ == "__main__":
    print("=" * 60)
    print("Topic Classification Dataset Loader Demo")
    print("=" * 60)
    
    loader = TopicDataLoader(vocab_size=5000, batch_size=8)
    train_loader, test_loader, vocab, samples = loader.load_topic_data()
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test a batch
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  labels: {batch['labels'].shape}")
    
    # Expected results
    expected = loader.get_expected_results()
    print(f"\nExpected Results:")
    print(f"  Task type: {expected.task_type}")
    print(f"  Dataset: {expected.dataset_name}")
    print(f"  Classes: {expected.num_classes}")
    print(f"  Class names: {expected.class_names}")
    print(f"  Sample inputs: {len(expected.sample_inputs)}")
