"""
Enhanced GPU Training & Evaluation Pipeline for Human Anomaly Detection (HAD)
Trains MobileNetV2 with fine-tuned top inverted residual blocks on ShanghaiTech dataset on GPU.
Evaluates on the 3,000 unseen test samples.
Outputs structured metrics to results/had/ (matching results/har/ structure).
"""

import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as tv_models
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("AURA.HAD.Train")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "human_anomaly_detection" / "dataset"
RESULTS_HAD_DIR = PROJECT_ROOT / "results" / "had"
MODELS_HAD_DIR = PROJECT_ROOT / "models" / "had"
LEGACY_MODELS_DIR = PROJECT_ROOT / "human_anomaly_detection" / "models"
LEGACY_RESULTS_DIR = PROJECT_ROOT / "human_anomaly_detection" / "results"


class ImprovedMobileNetV2(nn.Module):
    """
    MobileNetV2 with unfrozen top semantic layers and regularized dual-head.
    Low-level features remain frozen to preserve generalized ImageNet visual filters,
    while high-level blocks (12-18) are fine-tuned to learn human anomaly patterns.
    """
    def __init__(self, unfreeze_from_block: int = 12):
        super().__init__()
        backbone = tv_models.mobilenet_v2(weights=tv_models.MobileNet_V2_Weights.DEFAULT)
        self.features = backbone.features

        # Freeze early layers (0 to unfreeze_from_block - 1)
        for i, block in enumerate(self.features):
            if i < unfreeze_from_block:
                for p in block.parameters():
                    p.requires_grad = False
            else:
                for p in block.parameters():
                    p.requires_grad = True

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_paths(split: str):
    split_dir = DATASET_DIR / split
    norm_dir = split_dir / "normal"
    anom_dir = split_dir / "anomaly"

    paths, labels = [], []
    if norm_dir.exists():
        for f in sorted(os.listdir(norm_dir)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(str(norm_dir / f))
                labels.append(0)
    if anom_dir.exists():
        for f in sorted(os.listdir(anom_dir)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(str(anom_dir / f))
                labels.append(1)
    return paths, labels


def augment_img(img: np.ndarray) -> np.ndarray:
    if random.random() > 0.5:
        img = cv2.flip(img, 1)
    if random.random() > 0.5:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.85, 1.15), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    if random.random() > 0.5:
        angle = random.uniform(-8, 8)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return img


def load_batch(paths, labels, indices, is_training: bool = False):
    imgs = []
    targs = []
    for idx in indices:
        p = paths[idx]
        bgr = cv2.imread(p)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        if is_training:
            rgb = augment_img(rgb)
        normed = rgb.astype(np.float32) / 255.0
        imgs.append(normed)
        targs.append(labels[idx])
    return np.array(imgs, dtype=np.float32), np.array(targs, dtype=np.float32)


def plot_and_save_cm(cm, classes, out_path: Path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("HAD Confusion Matrix — Test Set (3,000 Samples)")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], "d"), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_and_save_curves(history, out_path: Path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r--s", label="Validation Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "b-o", label="Train Acc (%)")
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "r--s", label="Validation Acc (%)")
    ax2.set_title("Training & Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    if not torch.cuda.is_available():
        logger.error("CUDA is NOT available. This script is strictly configured to run on GPU only!")
        sys.exit(1)

    device = torch.device("cuda")
    logger.info(f"Target GPU acceleration verified: {torch.cuda.get_device_name(0)}")

    train_paths, train_labels = get_paths("train")
    val_paths, val_labels = get_paths("validation")
    test_paths, test_labels = get_paths("test")

    logger.info(f"Loaded ShanghaiTech Splits: Train={len(train_paths)}, Val={len(val_paths)}, Test={len(test_paths)}")

    # Model Initialization
    model = ImprovedMobileNetV2(unfreeze_from_block=12).to(device)

    # Differential Learning Rates
    backbone_params = [p for p in model.features.parameters() if p.requires_grad]
    head_params = list(model.classifier.parameters())
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": 1e-4},
        {"params": head_params, "lr": 5e-4},
    ], weight_decay=1e-4)

    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12, eta_min=1e-6)

    epochs = 12
    batch_size = 32
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 1
    best_weights_path = MODELS_HAD_DIR / "anomaly_model.pth"
    MODELS_HAD_DIR.mkdir(parents=True, exist_ok=True)

    n_train = len(train_paths)
    n_val = len(val_paths)
    train_indices = list(range(n_train))
    val_indices = list(range(n_val))

    start_time = time.time()
    logger.info("Starting GPU-accelerated fine-tuning loop...")

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        random.shuffle(train_indices)
        train_loss = 0.0
        train_correct = 0
        total_train_samples = 0

        for offset in range(0, n_train, batch_size):
            b_idx = train_indices[offset:offset + batch_size]
            bx, by = load_batch(train_paths, train_labels, b_idx, is_training=True)
            if len(bx) == 0:
                continue

            inputs = torch.from_numpy(bx).to(device)
            targets = torch.from_numpy(by).unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(bx)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            train_correct += (preds == targets).sum().item()
            total_train_samples += len(bx)

        scheduler.step()

        avg_train_loss = train_loss / total_train_samples
        avg_train_acc = train_correct / total_train_samples

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val_samples = 0

        with torch.no_grad():
            for offset in range(0, n_val, batch_size):
                b_idx = val_indices[offset:offset + batch_size]
                bx, by = load_batch(val_paths, val_labels, b_idx, is_training=False)
                if len(bx) == 0:
                    continue

                inputs = torch.from_numpy(bx).to(device)
                targets = torch.from_numpy(by).unsqueeze(1).to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * len(bx)
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                val_correct += (preds == targets).sum().item()
                total_val_samples += len(bx)

        avg_val_loss = val_loss / total_val_samples
        avg_val_acc = val_correct / total_val_samples

        history["train_loss"].append(round(avg_train_loss, 4))
        history["val_loss"].append(round(avg_val_loss, 4))
        history["train_acc"].append(round(avg_train_acc, 4))
        history["val_acc"].append(round(avg_val_acc, 4))

        logger.info(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc*100:.2f}% | Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc*100:.2f}%")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = avg_val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), best_weights_path)
            # Also mirror to legacy path
            LEGACY_MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), LEGACY_MODELS_DIR / "anomaly_model.pth")
            logger.info(f" --> Saved new best model checkpoint to {best_weights_path} (Val Loss: {best_val_loss:.4f})")

    duration = time.time() - start_time
    logger.info(f"GPU Training completed in {duration:.1f}s. Best Epoch: {best_epoch} with Val Acc: {best_val_acc*100:.2f}%")

    # Load Best Model for Final Unseen Test Evaluation
    logger.info("Evaluating optimal model on unseen TEST set (3,000 images)...")
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    model.eval()

    all_preds, all_probs, all_targets = [], [], []
    test_indices = list(range(len(test_paths)))

    with torch.no_grad():
        for offset in range(0, len(test_paths), batch_size):
            b_idx = test_indices[offset:offset + batch_size]
            bx, by = load_batch(test_paths, test_labels, b_idx, is_training=False)
            if len(bx) == 0:
                continue

            inputs = torch.from_numpy(bx).to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).squeeze(1).cpu().numpy()
            preds = (probs >= 0.50).astype(int)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(by)

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    acc = float(accuracy_score(all_targets, all_preds))
    prec = float(precision_score(all_targets, all_preds, average="macro"))
    rec = float(recall_score(all_targets, all_preds, average="macro"))
    f1_m = float(f1_score(all_targets, all_preds, average="macro"))
    f1_w = float(f1_score(all_targets, all_preds, average="weighted"))
    roc_auc = float(roc_auc_score(all_targets, all_probs))
    cm = confusion_matrix(all_targets, all_preds)
    cr_text = classification_report(all_targets, all_preds, target_names=["Normal", "Anomaly"], digits=4)

    logger.info(f"TEST METRICS: Accuracy: {acc*100:.2f}%, ROC-AUC: {roc_auc:.4f}, Macro F1: {f1_m*100:.2f}%")
    print(f"\nClassification Report:\n{cr_text}\n")

    # ── Populate results/had/ (Matching results/har/ Structure) ──────────────
    RESULTS_HAD_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_HAD_DIR / "classification_reports").mkdir(exist_ok=True)
    (RESULTS_HAD_DIR / "confusion_matrices").mkdir(exist_ok=True)
    (RESULTS_HAD_DIR / "training_curves").mkdir(exist_ok=True)

    # 1. Save classification reports
    with open(RESULTS_HAD_DIR / "classification_reports" / "had_classification_report.txt", "w") as f:
        f.write(cr_text)
    with open(LEGACY_RESULTS_DIR / "classification_report.txt", "w") as f:
        f.write(cr_text)

    # 2. Save confusion matrix plot
    plot_and_save_cm(cm, ["Normal", "Anomaly"], RESULTS_HAD_DIR / "confusion_matrices" / "had_confusion_matrix.png")
    plot_and_save_cm(cm, ["Normal", "Anomaly"], LEGACY_RESULTS_DIR / "confusion_matrix.png")

    # 3. Save training curve plot
    plot_and_save_curves(history, RESULTS_HAD_DIR / "training_curves" / "had_training_curve.png")
    plot_and_save_curves(history, LEGACY_RESULTS_DIR / "training_history.png")

    # 4. Save metrics.csv
    csv_line = "Model,Accuracy (%),Macro Precision (%),Macro Recall (%),Macro F1 (%),Weighted F1 (%),ROC-AUC,Architecture\n"
    csv_line += f"MobileNetV2 + DPGD,{acc*100:.2f},{prec*100:.2f},{rec*100:.2f},{f1_m*100:.2f},{f1_w*100:.2f},{roc_auc:.4f},MobileNetV2 Anomaly Classifier\n"
    with open(RESULTS_HAD_DIR / "metrics.csv", "w") as f:
        f.write(csv_line)

    # 5. Save had_results.json
    results_json = {
        "model_name": "MobileNetV2 Anomaly Classifier + DPGD",
        "accuracy": acc,
        "macro_precision": prec,
        "macro_recall": rec,
        "macro_f1": f1_m,
        "weighted_f1": f1_w,
        "roc_auc": roc_auc,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "confusion_matrix": cm.tolist(),
        "history": history,
        "training_duration_seconds": round(duration, 2)
    }
    with open(RESULTS_HAD_DIR / "had_results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    # Also update model_config.json
    config_dict = {
        "architecture": "mobilenet",
        "target_size": [224, 224],
        "calibrated_for_person_crops": True,
        "best_val_loss": round(best_val_loss, 4),
        "best_val_acc": round(best_val_acc, 4),
        "test_acc": round(acc, 4),
        "test_roc_auc": round(roc_auc, 4),
        "classes": {"0": "Normal", "1": "Anomaly"}
    }
    with open(MODELS_HAD_DIR / "model_config.json", "w") as f:
        json.dump(config_dict, f, indent=4)
    with open(LEGACY_MODELS_DIR / "model_config.json", "w") as f:
        json.dump(config_dict, f, indent=4)

    logger.info("All enhanced HAD results and model weights saved successfully!")


if __name__ == "__main__":
    main()
