"""
AURA Reliability ANN Evaluation and Baseline Benchmarking
Compares the trained Reliability ANN MLP against the baseline (direct YOLO confidence threshold).
Computes Accuracy, Precision, Recall, F1-Score, and ROC-AUC, and exports metrics.
"""

import os
import pickle
import argparse
import logging
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

from ann.model import ReliabilityANN
from ann.dataset import load_dataset_from_csv
from vision.dataset_collector import DatasetCollector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AURA.ANN.Evaluate")


def evaluate_baseline(
    confidences: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Evaluates the baseline YOLO confidence threshold classifier."""
    preds = (confidences >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, confidences) if len(np.unique(labels)) > 1 else 0.5),
    }


def evaluate_ann(
    model: nn.Module,
    X_scaled: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
    device: torch.device = torch.device("cpu"),
) -> Tuple[Dict[str, float], np.ndarray]:
    """Evaluates the neural network model."""
    model.eval()
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        probs = model(X_tensor).cpu().numpy().squeeze()
        
    # Handle single element array
    if probs.ndim == 0:
        probs = np.array([probs])

    preds = (probs >= threshold).astype(int)
    
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5),
    }
    return metrics, probs


def print_comparison_table(baseline_metrics: Dict[str, float], ann_metrics: Dict[str, float]) -> None:
    """Prints a comparison table of baseline vs ANN."""
    print("\n" + "=" * 65)
    print("      AURA RELIABILITY EVALUATION: BASELINE VS NEURAL NETWORK")
    print("=" * 65)
    print(f"| Metric       | Baseline (YOLO Conf) | Reliability ANN | Improvement |")
    print(f"|--------------|----------------------|-----------------|-------------|")
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        b_val = baseline_metrics[key]
        a_val = ann_metrics[key]
        diff = a_val - b_val
        sign = "+" if diff >= 0 else ""
        print(f"| {key:<12} | {b_val:.4f}               | {a_val:.4f}          | {sign}{diff:.4f}      |")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate AURA Reliability ANN vs Baseline")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/reliability_ann.pth",
        help="Path to trained model weights (.pth). Default: 'models/reliability_ann.pth'",
    )
    parser.add_argument(
        "--scaler-path",
        type=str,
        default="models/scaler.pkl",
        help="Path to fitted scaler metadata (.pkl). Default: 'models/scaler.pkl'",
    )
    parser.add_argument(
        "--dataset-csv",
        type=str,
        default=None,
        help="Path to CSV dataset. If not provided, a synthetic dataset is generated.",
    )
    parser.add_argument(
        "--baseline-thresh",
        type=float,
        default=0.5,
        help="Confidence threshold for baseline. Default: 0.5",
    )
    parser.add_argument(
        "--ann-thresh",
        type=float,
        default=0.5,
        help="Reliability threshold for ANN decision. Default: 0.5",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use ('auto', 'cpu', 'cuda'). Default: 'auto'",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Check for saved artifacts
    if not os.path.exists(args.model_path) or not os.path.exists(args.scaler_path):
        logger.error("Trained model weights or scaler not found. Please run training first.")
        logger.info("Example command: python -m ann.train")
        return

    # Load scaler metadata
    with open(args.scaler_path, "rb") as f:
        scaler_metadata = pickle.load(f)
    
    scaler = scaler_metadata["scaler"]
    include_class_onehot = scaler_metadata.get("include_class_onehot", False)
    feature_family = scaler_metadata.get("feature_family", "all")
    input_dim = scaler_metadata.get("input_dim", 14)

    # Load / Generate dataset
    if args.dataset_csv:
        logger.info(f"Loading evaluation dataset from CSV: '{args.dataset_csv}'...")
        X_raw, y = load_dataset_from_csv(
            filepath=args.dataset_csv,
            include_class_onehot=include_class_onehot,
            feature_family=feature_family,
        )
    else:
        logger.info("No CSV dataset path provided. Generating 300 synthetic evaluation samples...")
        collector = DatasetCollector.generate_mock_dataset(num_samples=300, seed=args.seed + 1)
        X_full, y = collector.to_arrays(include_class_onehot=include_class_onehot)
        
        # Slicing feature columns to match family
        if feature_family == "geometry":
            X_raw = X_full[:, :11]
        elif feature_family == "quality":
            X_raw = X_full[:, 11:14]
        else:
            X_raw = X_full

    # Pre-process raw features using the loaded training scaler
    X_scaled = scaler.transform(X_raw)

    # Get confidence scores from the raw features (always the first feature)
    # Note: if feature family is 'quality', confidence won't be in X_raw.
    # We load standard collector to get raw confidence for baseline.
    if feature_family == "quality":
        # Need raw confidence specifically for the baseline comparison
        if args.dataset_csv:
            df = pd.read_csv(args.dataset_csv)
            confidences = df["confidence"].values.astype(np.float32)
        else:
            confidences = X_full[:, 0]
    else:
        confidences = X_raw[:, 0]

    # Load model
    model = ReliabilityANN(input_dim=input_dim)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    # Evaluate Baseline
    baseline_metrics = evaluate_baseline(confidences, y, threshold=args.baseline_thresh)

    # Evaluate ANN
    ann_metrics, probs = evaluate_ann(model, X_scaled, y, threshold=args.ann_thresh, device=device)

    # Print comparison
    print_comparison_table(baseline_metrics, ann_metrics)


if __name__ == "__main__":
    main()
