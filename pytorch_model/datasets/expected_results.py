"""
Expected Results Schema for Dataset Evaluation.

Defines the structure for expected results across different dataset types
and provides utilities for validation.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import json


class TaskType(Enum):
    """Supported task types."""
    CLASSIFICATION = "classification"
    QA = "qa"
    REGRESSION = "regression"
    MULTIPLE_CHOICE = "multiple_choice"


class DatasetType(Enum):
    """Dataset source types."""
    IMDB = "imdb"
    TOPIC = "topic"
    SQuAD = "squad"
    CUSTOM = "custom"
    SYNTHETIC = "synthetic"


@dataclass
class ExpectedResults:
    """
    Schema for expected results in dataset evaluation.
    
    Attributes:
        task_type: Type of task (classification, qa, regression)
        dataset_name: Name of the dataset
        num_classes: Number of classes (for classification)
        class_names: Names of each class
        sample_inputs: List of sample input texts
        expected_labels: Ground truth labels for samples
        expected_probabilities: Optional expected probability distributions
        description: Human-readable description
        metadata: Additional dataset metadata
    """
    task_type: str = "classification"
    dataset_name: str = ""
    num_classes: int = 2
    class_names: List[str] = field(default_factory=list)
    sample_inputs: List[str] = field(default_factory=list)
    expected_labels: List[int] = field(default_factory=list)
    expected_probabilities: Optional[List[List[float]]] = None
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate after initialization."""
        if len(self.sample_inputs) != len(self.expected_labels):
            raise ValueError(
                f"sample_inputs ({len(self.sample_inputs)}) must have same length "
                f"as expected_labels ({len(self.expected_labels)})"
            )
        
        if self.task_type not in [t.value for t in TaskType]:
            raise ValueError(f"Invalid task_type: {self.task_type}")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ExpectedResults":
        """Create from dictionary."""
        return cls(**data)
    
    def save(self, path: str):
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "ExpectedResults":
        """Load from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def validate_predictions(
        self,
        predictions: List[int],
        probabilities: Optional[List[List[float]]] = None
    ) -> Dict[str, Any]:
        """
        Validate predictions against expected results.
        
        Args:
            predictions: List of predicted class indices
            probabilities: Optional list of probability distributions
        
        Returns:
            Dictionary with validation results
        """
        if len(predictions) != len(self.expected_labels):
            return {
                "valid": False,
                "error": f"Length mismatch: {len(predictions)} vs {len(self.expected_labels)}"
            }
        
        # Calculate accuracy
        correct = sum(1 for p, e in zip(predictions, self.expected_labels) if p == e)
        accuracy = correct / len(predictions) if predictions else 0.0
        
        # Per-class analysis
        per_class = {}
        for class_idx, class_name in enumerate(self.class_names):
            class_mask = [i for i, e in enumerate(self.expected_labels) if e == class_idx]
            if class_mask:
                class_correct = sum(
                    1 for i in class_mask if predictions[i] == self.expected_labels[i]
                )
                per_class[class_name] = {
                    "total": len(class_mask),
                    "correct": class_correct,
                    "accuracy": class_correct / len(class_mask)
                }
        
        # Validate probability sums if provided
        prob_valid = True
        if probabilities:
            for i, probs in enumerate(probabilities):
                if not abs(sum(probs) - 1.0) < 1e-5:
                    prob_valid = False
                    break
        
        return {
            "valid": True,
            "accuracy": accuracy,
            "correct": correct,
            "total": len(predictions),
            "per_class": per_class,
            "probabilities_valid": prob_valid,
        }


@dataclass
class IMDBExpectedResults(ExpectedResults):
    """Expected results specifically for IMDB dataset."""
    
    def __init__(self):
        super().__init__(
            task_type="classification",
            dataset_name="IMDB",
            num_classes=2,
            class_names=["negative", "positive"],
            description="IMDB Movie Review Sentiment Classification"
        )


@dataclass  
class TopicExpectedResults(ExpectedResults):
    """Expected results specifically for Topic Classification dataset."""
    
    def __init__(
        self,
        num_classes: int = 4,
        class_names: Optional[List[str]] = None
    ):
        if class_names is None:
            class_names = ["Science", "Sports", "Business", "Technology"]
        
        super().__init__(
            task_type="classification",
            dataset_name="TopicClassification",
            num_classes=num_classes,
            class_names=class_names,
            description="Topic Classification with Q&A Format"
        )


@dataclass
class QAExpectedResults(ExpectedResults):
    """Expected results for Question Answering tasks."""
    
    def __init__(
        self,
        dataset_name: str = "QA",
        sample_questions: Optional[List[str]] = None,
        sample_contexts: Optional[List[str]] = None,
        expected_answers: Optional[List[str]] = None,
    ):
        super().__init__(
            task_type="qa",
            dataset_name=dataset_name,
            description="Question Answering"
        )
        self.sample_questions = sample_questions or []
        self.sample_contexts = sample_contexts or []
        self.expected_answers = expected_answers or []


class ExpectedResultsRegistry:
    """
    Registry for managing expected results across datasets.
    
    Example:
        >>> registry = ExpectedResultsRegistry()
        >>> registry.register("imdb", IMDBExpectedResults())
        >>> results = registry.get("imdb")
    """
    
    def __init__(self):
        self._registry: Dict[str, ExpectedResults] = {}
        self._register_defaults()
    
    def _register_defaults(self):
        """Register default expected results."""
        self.register("imdb", IMDBExpectedResults())
        self.register("topic", TopicExpectedResults())
    
    def register(self, name: str, results: ExpectedResults):
        """Register expected results for a dataset."""
        self._registry[name] = results
    
    def get(self, name: str) -> Optional[ExpectedResults]:
        """Get expected results by dataset name."""
        return self._registry.get(name)
    
    def list_datasets(self) -> List[str]:
        """List all registered datasets."""
        return list(self._registry.keys())
    
    def save_all(self, directory: str):
        """Save all expected results to a directory."""
        import os
        os.makedirs(directory, exist_ok=True)
        
        for name, results in self._registry.items():
            path = os.path.join(directory, f"{name}_expected.json")
            results.save(path)
    
    def load_all(self, directory: str):
        """Load all expected results from a directory."""
        import os
        from pathlib import Path
        
        for path in Path(directory).glob("*_expected.json"):
            name = path.stem.replace("_expected", "")
            results = ExpectedResults.load(str(path))
            self.register(name, results)


# Utility functions
def create_synthetic_expected_results(
    num_samples: int = 100,
    num_classes: int = 4,
) -> ExpectedResults:
    """
    Create expected results for synthetic dataset.
    
    Args:
        num_samples: Number of samples to generate
        num_classes: Number of classes
    
    Returns:
        ExpectedResults with synthetic data
    """
    class_names = [f"Class_{i}" for i in range(num_classes)]
    
    sample_inputs = [f"Synthetic input {i}" for i in range(num_samples)]
    expected_labels = [i % num_classes for i in range(num_samples)]
    
    return ExpectedResults(
        task_type="classification",
        dataset_name="Synthetic",
        num_classes=num_classes,
        class_names=class_names,
        sample_inputs=sample_inputs,
        expected_labels=expected_labels,
        description="Synthetic classification dataset",
        metadata={"source": "generated"}
    )


def validate_dataset_format(dataset: Any, expected: ExpectedResults) -> Dict:
    """
    Validate that a dataset matches the expected format.
    
    Args:
        dataset: The dataset to validate
        expected: Expected results schema
    
    Returns:
        Validation results dictionary
    """
    results = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Check if dataset has required methods
    if not hasattr(dataset, '__len__'):
        results["valid"] = False
        results["errors"].append("Dataset missing __len__ method")
    
    if not hasattr(dataset, '__getitem__'):
        results["valid"] = False
        results["errors"].append("Dataset missing __getitem__ method")
    
    # Check expected results schema
    if expected.num_classes < 2:
        results["warnings"].append(f"num_classes={expected.num_classes} seems too low")
    
    if not expected.class_names:
        results["warnings"].append("class_names is empty")
    
    return results


# Sample usage
if __name__ == "__main__":
    print("=" * 60)
    print("Expected Results Schema Demo")
    print("=" * 60)
    
    # Create IMDB expected results
    imdb_results = IMDBExpectedResults()
    print(f"\nIMDB Results: {imdb_results.to_dict()}")
    
    # Create Topic expected results
    topic_results = TopicExpectedResults(num_classes=4)
    print(f"\nTopic Results: {topic_results.to_dict()}")
    
    # Test validation
    predictions = [0, 1, 1, 0, 1]
    expected_labels = [0, 1, 0, 0, 1]
    
    topic_results.sample_inputs = ["q1", "q2", "q3", "q4", "q5"]
    topic_results.expected_labels = expected_labels
    
    validation = topic_results.validate_predictions(predictions)
    print(f"\nValidation: {validation}")
    
    # Test registry
    registry = ExpectedResultsRegistry()
    print(f"\nRegistered datasets: {registry.list_datasets()}")
