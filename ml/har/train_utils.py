"""
AURA HAR Deep Learning Training Utility
Provides standard training loop with:
- Train / Validation splitting (derived ONLY from official training set)
- Early stopping & ReduceLROnPlateau
- Best model checkpointing
- Training curve logging (loss & accuracy)
- Evaluation on official UCF101 test set
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

logger = logging.getLogger("AURA.HAR.TrainUtils")


def get_device() -> torch.device:
    """Returns CUDA device if available, else CPU fallback."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using GPU Acceleration: %s", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logger.info("Using CPU fallback for training.")
    return device


def prepare_dataloaders(
    data_dir: Path,
    batch_size: int = 32,
    val_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Loads data and creates Train, Validation (from train only), and Test DataLoaders.
    Strictly avoids data leakage: test set is NEVER used for validation.
    """
    X_train_raw = np.load(data_dir / "X_train.npy").astype(np.float32)
    y_train_raw = np.load(data_dir / "y_train.npy").astype(np.int64)
    X_test = np.load(data_dir / "X_test.npy").astype(np.float32)
    y_test = np.load(data_dir / "y_test.npy").astype(np.int64)

    with open(data_dir / "dataset_meta.json", "r") as f:
        meta = json.load(f)

    # Stratified validation split derived ONLY from training data
    train_idx, val_idx = train_test_split(
        np.arange(len(y_train_raw)),
        test_size=val_ratio,
        stratify=y_train_raw,
        random_state=random_seed
    )

    X_train = X_train_raw[train_idx]
    y_train = y_train_raw[train_idx]
    X_val = X_train_raw[val_idx]
    y_val = y_train_raw[val_idx]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    split_info = {
        "num_train": len(X_train),
        "num_val": len(X_val),
        "num_test": len(X_test),
        "classes": meta["classes"],
    }
    return train_loader, val_loader, test_loader, split_info


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    model_save_path: Path,
    model_name: str,
    epochs: int = 60,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 15,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Standard training loop with Early Stopping, Checkpointing, and Evaluation.
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
    }

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # 1. Training Phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(batch_y)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == batch_y).sum().item()
            total += len(batch_y)

        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total

        # 2. Validation Phase (only from train-val split)
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item() * len(batch_y)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_total += len(batch_y)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        scheduler.step(epoch_val_acc)

        history["train_loss"].append(round(epoch_train_loss, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["train_acc"].append(round(epoch_train_acc, 4))
        history["val_acc"].append(round(epoch_val_acc, 4))

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                f"[{model_name} Epoch {epoch:02d}/{epochs}] "
                f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc*100:.1f}% | "
                f"Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc*100:.1f}%"
            )

        # Early Stopping & Best Checkpoint
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}. Best Val Acc: {best_val_acc*100:.2f}% (Epoch {best_epoch})")
                break

    # 3. Load Best Model and Evaluate on Official UCF101 Test Set
    logger.info(f"Loading best checkpoint from {model_save_path} for test set evaluation...")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(batch_y.numpy())

    y_test = np.array(all_targets)
    y_pred = np.array(all_preds)

    acc = float(accuracy_score(y_test, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    logger.info("=" * 60)
    logger.info(f"{model_name.upper()} TEST SET RESULTS:")
    logger.info(f"Accuracy:       {acc * 100:.2f}%")
    logger.info(f"Macro F1:       {f1_macro * 100:.2f}%")
    logger.info(f"Weighted F1:    {f1_weight * 100:.2f}%")
    logger.info("=" * 60)

    return {
        "model_name": model_name,
        "accuracy": acc,
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weight),
        "best_epoch": best_epoch,
        "best_val_acc": float(best_val_acc),
        "confusion_matrix": cm.tolist(),
        "history": history,
        "y_true": y_test.tolist(),
        "y_pred": y_pred.tolist(),
    }
