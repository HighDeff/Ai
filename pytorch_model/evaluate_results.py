"""
Evaluation Results Reporter for Dataset Evaluation.

Generates structured comparison of predictions vs expected results,
including accuracy metrics, per-class breakdown, and misclassified examples.

Example:
    >>> reporter = ResultsReporter("IMDB", "classification")
    >>> reporter.add_prediction("Great movie!", expected=1, predicted=1, confidence=0.95)
    >>> results = reporter.generate_report()
    >>> reporter.save_to_json("results/report.json")
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))

from datasets.expected_results import ExpectedResults


@dataclass
class PredictionRecord:
    """Single prediction record."""
    input_text: str
    expected_label: int
    predicted_label: int
    confidence: float
    probabilities: List[float]
    correct: bool
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ClassMetrics:
    """Per-class metrics."""
    precision: float
    recall: float
    f1: float
    support: int
    true_positives: int
    false_positives: int
    false_negatives: int


class ResultsReporter:
    """
    Reporter for tracking and analyzing prediction results.
    
    Example:
        >>> reporter = ResultsReporter("IMDB", "classification")
        >>> reporter.add_prediction(input_text, expected=1, predicted=1, confidence=0.95)
        >>> reporter.add_prediction(input_text, expected=0, predicted=1, confidence=0.85)
        >>> report = reporter.generate_report()
        >>> print(f"Accuracy: {report['accuracy']:.2%}")
    """
    
    def __init__(
        self,
        dataset_name: str,
        task_type: str,
        class_names: Optional[List[str]] = None,
        num_classes: int = 2,
    ):
        self.dataset_name = dataset_name
        self.task_type = task_type
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.num_classes = num_classes
        
        self.records: List[PredictionRecord] = []
        self.metadata: Dict[str, Any] = {
            "start_time": datetime.now().isoformat(),
            "end_time": None,
        }
    
    def add_prediction(
        self,
        input_text: str,
        expected_label: int,
        predicted_label: int,
        confidence: float,
        probabilities: Optional[List[float]] = None,
    ):
        """
        Record a prediction result.
        
        Args:
            input_text: The input text that was classified
            expected_label: Ground truth label
            predicted_label: Model's predicted label
            confidence: Confidence score (probability of predicted class)
            probabilities: Full probability distribution
        """
        if probabilities is None:
            probabilities = [0.0] * self.num_classes
            probabilities[predicted_label] = confidence
        
        record = PredictionRecord(
            input_text=input_text,
            expected_label=expected_label,
            predicted_label=predicted_label,
            confidence=confidence,
            probabilities=probabilities,
            correct=(expected_label == predicted_label),
        )
        
        self.records.append(record)
    
    def add_batch(
        self,
        texts: List[str],
        expected_labels: torch.Tensor,
        predictions: torch.Tensor,
        probabilities: torch.Tensor,
    ):
        """
        Add a batch of predictions at once.
        
        Args:
            texts: List of input texts
            expected_labels: Ground truth labels tensor
            predictions: Predicted labels tensor
            probabilities: Probability distribution tensor
        """
        expected_labels = expected_labels.cpu().numpy()
        predictions = predictions.cpu().numpy()
        probabilities = probabilities.cpu().numpy()
        
        for i in range(len(texts)):
            confidence = probabilities[i][predictions[i]]
            self.add_prediction(
                input_text=texts[i],
                expected_label=int(expected_labels[i]),
                predicted_label=int(predictions[i]),
                confidence=float(confidence),
                probabilities=probabilities[i].tolist(),
            )
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report.
        
        Returns:
            Dictionary containing accuracy metrics, per-class breakdown,
            and misclassified examples
        """
        if not self.records:
            return {"error": "No predictions recorded"}
        
        self.metadata["end_time"] = datetime.now().isoformat()
        
        # Overall accuracy
        total = len(self.records)
        correct = sum(1 for r in self.records if r.correct)
        accuracy = correct / total if total > 0 else 0.0
        
        # Per-class metrics
        per_class = self._calculate_per_class_metrics()
        
        # Confusion matrix
        confusion_matrix = self._build_confusion_matrix()
        
        # Misclassified examples
        misclassified = self._get_misclassified_examples(limit=20)
        
        # Confidence statistics
        confidence_stats = self._calculate_confidence_stats()
        
        return {
            "metadata": self.metadata,
            "dataset_name": self.dataset_name,
            "task_type": self.task_type,
            "num_samples": total,
            "overall": {
                "accuracy": accuracy,
                "correct": correct,
                "incorrect": total - correct,
            },
            "per_class": per_class,
            "confusion_matrix": confusion_matrix,
            "misclassified_examples": misclassified,
            "confidence_stats": confidence_stats,
        }
    
    def _calculate_per_class_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate precision, recall, F1 for each class."""
        metrics = {}
        
        for class_idx in range(self.num_classes):
            class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f"class_{class_idx}"
            
            # Count TP, FP, FN
            tp = sum(1 for r in self.records if r.predicted_label == class_idx and r.expected_label == class_idx)
            fp = sum(1 for r in self.records if r.predicted_label == class_idx and r.expected_label != class_idx)
            fn = sum(1 for r in self.records if r.predicted_label != class_idx and r.expected_label == class_idx)
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics[class_name] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": tp + fn,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
            }
        
        # Macro averages
        all_precisions = [m["precision"] for m in metrics.values()]
        all_recalls = [m["recall"] for m in metrics.values()]
        all_f1s = [m["f1"] for m in metrics.values()]
        
        metrics["macro_avg"] = {
            "precision": sum(all_precisions) / len(all_precisions),
            "recall": sum(all_recalls) / len(all_recalls),
            "f1": sum(all_f1s) / len(all_f1s),
        }
        
        return metrics
    
    def _build_confusion_matrix(self) -> List[List[int]]:
        """Build confusion matrix."""
        matrix = [[0] * self.num_classes for _ in range(self.num_classes)]
        
        for r in self.records:
            matrix[r.expected_label][r.predicted_label] += 1
        
        return matrix
    
    def _get_misclassified_examples(self, limit: int = 20) -> List[Dict]:
        """Get examples that were misclassified."""
        misclassified = []
        
        for r in self.records:
            if not r.correct:
                expected_name = self.class_names[r.expected_label] if r.expected_label < len(self.class_names) else f"class_{r.expected_label}"
                predicted_name = self.class_names[r.predicted_label] if r.predicted_label < len(self.class_names) else f"class_{r.predicted_label}"
                
                misclassified.append({
                    "input_text": r.input_text[:200],  # Truncate long texts
                    "expected_label": r.expected_label,
                    "expected_name": expected_name,
                    "predicted_label": r.predicted_label,
                    "predicted_name": predicted_name,
                    "confidence": r.confidence,
                    "timestamp": r.timestamp,
                })
                
                if len(misclassified) >= limit:
                    break
        
        return misclassified
    
    def _calculate_confidence_stats(self) -> Dict[str, float]:
        """Calculate confidence score statistics."""
        confidences = [r.confidence for r in self.records]
        
        if not confidences:
            return {}
        
        sorted_conf = sorted(confidences)
        n = len(sorted_conf)
        
        return {
            "mean": sum(confidences) / n,
            "min": min(confidences),
            "max": max(confidences),
            "median": sorted_conf[n // 2],
            "std": self._calculate_std(confidences),
        }
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def save_to_json(self, path: str):
        """
        Save full report to JSON file.
        
        Args:
            path: Path to save the JSON file
        """
        report = self.generate_report()
        
        # Convert PredictionRecords to dicts
        report["prediction_records"] = [
            {
                "input_text": r.input_text,
                "expected_label": r.expected_label,
                "predicted_label": r.predicted_label,
                "confidence": r.confidence,
                "probabilities": r.probabilities,
                "correct": r.correct,
                "timestamp": r.timestamp,
            }
            for r in self.records
        ]
        
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to {path}")
    
    def save_summary(self, path: str):
        """Save a summary report (without full prediction records)."""
        report = self.generate_report()
        
        # Remove heavy data
        if "prediction_records" in report:
            del report["prediction_records"]
        
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Summary saved to {path}")
    
    def print_summary(self):
        """Print a human-readable summary."""
        report = self.generate_report()
        
        print("\n" + "=" * 60)
        print(f"EVALUATION REPORT: {self.dataset_name}")
        print("=" * 60)
        
        print(f"\nOverall Accuracy: {report['overall']['accuracy']:.2%}")
        print(f"Correct: {report['overall']['correct']} / {report['num_samples']}")
        
        print("\nPer-Class Metrics:")
        print("-" * 40)
        
        for class_name, metrics in report["per_class"].items():
            if class_name == "macro_avg":
                continue
            print(f"\n{class_name}:")
            print(f"  Precision: {metrics['precision']:.2%}")
            print(f"  Recall:    {metrics['recall']:.2%}")
            print(f"  F1:        {metrics['f1']:.2%}")
            print(f"  Support:   {metrics['support']}")
        
        if "macro_avg" in report["per_class"]:
            print(f"\nMacro Average F1: {report['per_class']['macro_avg']['f1']:.2%}")
        
        print("\nConfidence Statistics:")
        print("-" * 40)
        stats = report["confidence_stats"]
        print(f"  Mean:     {stats['mean']:.3f}")
        print(f"  Median:   {stats['median']:.3f}")
        print(f"  Min:      {stats['min']:.3f}")
        print(f"  Max:      {stats['max']:.3f}")
        print(f"  Std Dev:  {stats['std']:.3f}")
        
        print("\n" + "=" * 60)


def run_evaluation(
    model,
    data_loader,
    vocab,
    expected_results: ExpectedResults,
    device: str = "cpu",
) -> ResultsReporter:
    """
    Run evaluation on a dataset and generate report.
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for the dataset
        vocab: Vocabulary for decoding
        expected_results: Expected results schema
        device: Device to run on
    
    Returns:
        ResultsReporter with evaluation results
    """
    reporter = ResultsReporter(
        dataset_name=expected_results.dataset_name,
        task_type=expected_results.task_type,
        class_names=expected_results.class_names,
        num_classes=expected_results.num_classes,
    )
    
    model.eval()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Get predictions
            outputs = model(input_ids)
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
            
            # Decode texts (if available)
            texts = []
            for ids in input_ids:
                text = vocab.decode(ids.tolist())
                texts.append(text[:100])  # Truncate for report
            
            # Add batch to reporter
            reporter.add_batch(
                texts=texts,
                expected_labels=labels,
                predictions=predictions,
                probabilities=probs,
            )
    
    return reporter


# Sample usage
if __name__ == "__main__":
    print("=" * 60)
    print("Results Reporter Demo")
    print("=" * 60)
    
    # Create reporter
    reporter = ResultsReporter(
        dataset_name="IMDB",
        task_type="classification",
        class_names=["negative", "positive"],
        num_classes=2,
    )
    
    # Add sample predictions
    test_cases = [
        ("This movie was amazing!", 1, 1, 0.95),
        ("Terrible film, waste of time", 0, 0, 0.88),
        ("It was okay, nothing special", 1, 0, 0.65),  # Wrong
        ("Best movie ever made!", 1, 1, 0.92),
        ("I hated every minute of it", 0, 0, 0.91),
        ("Not bad, actually quite good", 1, 1, 0.72),
        ("Completely boring and dull", 0, 0, 0.85),
        ("Loved the acting, great story", 1, 1, 0.89),
    ]
    
    for text, expected, predicted, confidence in test_cases:
        reporter.add_prediction(text, expected, predicted, confidence)
    
    # Generate and print report
    reporter.print_summary()
    
    # Save to file
    print("\nSaving detailed report...")
    reporter.save_to_json("results/detailed_report.json")
    reporter.save_summary("results/summary.json")
    
    print("\nDemo complete!")
