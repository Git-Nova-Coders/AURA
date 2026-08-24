"""
AURA Reliability ANN Training Pipeline
Trains the custom Reliability ANN MLP using binary cross-entropy loss.
Saves model weights and the feature normalization scaler to the models directory.
"""

import os
import pickle
import random
import argparse
import logging
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ann.model import ReliabilityANN
from ann.dataset import load_dataset_from_csv, prepare_data_loaders
from vision.dataset_collector import DatasetCollector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AURA.ANN.Train")


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int = 50,
    lr: float = 0.01,
    device: torch.device = torch.device("cpu"),
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Trains the PyTorch model and tracks metrics."""
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1": [],
    }

    logger.info(f"Starting model training for {epochs} epochs on device '{device}'...")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                
                preds = (outputs >= 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, zero_division=0)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["val_f1"].append(val_f1)

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            logger.info(
                f"Epoch {epoch:2d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f} | Val F1: {val_f1:.4f}"
            )

    return model, history


def evaluate_test_set(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Runs evaluation on the test split."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = (outputs >= 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    return {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
        "recall": float(recall_score(all_labels, all_preds, zero_division=0)),
        "f1": float(f1_score(all_labels, all_preds, zero_division=0)),
    }


def main():
    parser = argparse.ArgumentParser(description="Train AURA Detection Reliability ANN")
    parser.add_argument(
        "--dataset-csv",
        type=str,
        default=None,
        help="Path to CSV dataset. If not provided, a synthetic dataset is generated.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs to train. Default: 50",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size. Default: 32",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate. Default: 0.01",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Hardware device ('auto', 'cpu', 'cuda'). Default: 'auto'",
    )
    parser.add_argument(
        "--out-model-path",
        type=str,
        default="models/reliability_ann.pth",
        help="Output path for PyTorch weights (.pth). Default: 'models/reliability_ann.pth'",
    )
    parser.add_argument(
        "--out-scaler-path",
        type=str,
        default="models/scaler.pkl",
        help="Output path for fitted scaler (.pkl). Default: 'models/scaler.pkl'",
    )
    parser.add_argument(
        "--include-class-onehot",
        action="store_true",
        help="Whether to include class one-hot feature encoding in input vector.",
    )
    parser.add_argument(
        "--feature-family",
        type=str,
        default="all",
        choices=["all", "geometry", "quality"],
        help="Feature family: 'all' (14 features), 'geometry' (11 features), 'quality' (3 features). Default: 'all'",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42",
    )
    args = parser.parse_args()

    # Set seeds
    set_seed(args.seed)

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load / Generate dataset
    if args.dataset_csv:
        logger.info(f"Loading dataset from CSV: '{args.dataset_csv}'...")
        try:
            X, y = load_dataset_from_csv(
                filepath=args.dataset_csv,
                include_class_onehot=args.include_class_onehot,
                feature_family=args.feature_family,
            )
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return
    else:
        logger.info("No CSV dataset path provided. Generating 1000 synthetic samples for training...")
        collector = DatasetCollector.generate_mock_dataset(num_samples=1000, seed=args.seed)
        # Handle feature family filters for arrays
        X_full, y_full = collector.to_arrays(include_class_onehot=args.include_class_onehot)
        
        # Slicing feature columns to match family
        if args.feature_family == "geometry":
            X = X_full[:, :11]  # BBox geometry only
        elif args.feature_family == "quality":
            X = X_full[:, 11:14]  # quality features only
        else:
            X = X_full  # All 14 features
        y = y_full

    input_dim = X.shape[1]
    logger.info(f"Loaded {len(X)} samples with input feature dimension = {input_dim}")

    # Build DataLoaders
    train_loader, val_loader, test_loader, scaler = prepare_data_loaders(
        X=X,
        y=y,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # Initialize model
    model = ReliabilityANN(input_dim=input_dim)

    # Train model
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
    )

    # Evaluate on test set
    test_metrics = evaluate_test_set(model, test_loader, device)
    logger.info("=" * 60)
    logger.info("Reliability ANN Test Set Evaluation:")
    logger.info(f"  Accuracy  : {test_metrics['accuracy']:.4f}")
    logger.info(f"  Precision : {test_metrics['precision']:.4f}")
    logger.info(f"  Recall    : {test_metrics['recall']:.4f}")
    logger.info(f"  F1-Score  : {test_metrics['f1']:.4f}")
    logger.info("=" * 60)

    # Save artifacts
    os.makedirs(os.path.dirname(os.path.abspath(args.out_model_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_scaler_path)), exist_ok=True)

    # Save weights
    torch.save(model.state_dict(), args.out_model_path)
    logger.info(f"Saved PyTorch model weights to '{args.out_model_path}'")

    # Save scaler and pre-processing config
    scaler_metadata = {
        "scaler": scaler,
        "input_dim": input_dim,
        "include_class_onehot": args.include_class_onehot,
        "feature_family": args.feature_family,
        "model_version": "ann_v1",
    }
    with open(args.out_scaler_path, "wb") as f:
        pickle.dump(scaler_metadata, f)
    logger.info(f"Saved pre-processing scaler metadata to '{args.out_scaler_path}'")


if __name__ == "__main__":
    main()
