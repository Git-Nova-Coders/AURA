"""
AURA HAR Phase 7: Unified Model Comparison & Scientific Evaluation
Aggregates and compares:
1. Random Forest Baseline
2. Spatial Pose CNN
3. Temporal Pose LSTM
4. Hybrid Pose CNN + LSTM

Generates:
- metrics.csv (Master comparison table: Accuracy, Macro P/R/F1, Weighted F1, Params)
- classification_reports/ (Per-class precision, recall, f1 for all models)
- confusion_matrices/ (Normalized confusion matrix plots)
- training_curves/ (Loss and Accuracy training/validation convergence plots)
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("AURA.HAR.Evaluate")

RESULTS_DIR = PROJECT_ROOT / "results" / "har"
DATA_DIR = PROJECT_ROOT / "data" / "har" / "processed"


def plot_confusion_matrix(cm: np.ndarray, classes: List[str], model_name: str, save_path: Path):
    """Generates a styled confusion matrix plot using pure matplotlib."""
    plt.figure(figsize=(9, 7))
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    im = plt.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    plt.title(f"Normalized Confusion Matrix — {model_name}", fontsize=13, pad=14, fontweight="bold")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right", fontsize=9)
    plt.yticks(tick_marks, classes, fontsize=9)

    # Annotate cells with values
    thresh = cm_norm.max() / 2.0
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            val = cm_norm[i, j]
            color = "white" if val > thresh else "black"
            text = f"{val:.2f}\n({cm[i, j]})" if cm[i, j] > 0 else "0.00"
            plt.text(j, i, text, ha="center", va="center", color=color, fontsize=8)

    plt.ylabel("Ground Truth Class", fontsize=11, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=250)
    plt.close()


def plot_training_curves(history: Dict[str, List[float]], model_name: str, save_path: Path):
    """Generates loss and accuracy convergence plots."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#2563eb", linewidth=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#dc2626", linestyle="--", linewidth=2)
    ax1.set_title(f"{model_name} - Loss Curve")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("CrossEntropy Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Acc", color="#16a34a", linewidth=2)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], label="Val Acc", color="#f59e0b", linestyle="--", linewidth=2)
    ax2.set_title(f"{model_name} - Accuracy Curve (%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def generate_unified_evaluation():
    """Compiles all model evaluation results into comparative reports and figures."""
    logger.info("=" * 60)
    logger.info("PHASE 7: UNIFIED MODEL EVALUATION & SCIENTIFIC COMPARISON")
    logger.info("=" * 60)

    with open(DATA_DIR / "dataset_meta.json", "r") as f:
        meta = json.load(f)
    classes = meta["classes"]

    cm_dir = RESULTS_DIR / "confusion_matrices"
    curves_dir = RESULTS_DIR / "training_curves"
    reports_dir = RESULTS_DIR / "classification_reports"
    cm_dir.mkdir(parents=True, exist_ok=True)
    curves_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    records = []

    # 1. Random Forest Baseline
    rf_report_path = reports_dir / "random_forest_report.json"
    rf_cm_path = cm_dir / "random_forest_cm.npy"
    if rf_report_path.exists() and rf_cm_path.exists():
        with open(rf_report_path, "r") as f:
            rf_rep = json.load(f)
        rf_cm = np.load(rf_cm_path)
        records.append({
            "Model": "Random Forest Baseline",
            "Accuracy (%)": round(rf_rep["accuracy"] * 100, 2),
            "Macro Precision (%)": round(rf_rep["macro avg"]["precision"] * 100, 2),
            "Macro Recall (%)": round(rf_rep["macro avg"]["recall"] * 100, 2),
            "Macro F1 (%)": round(rf_rep["macro avg"]["f1-score"] * 100, 2),
            "Weighted F1 (%)": round(rf_rep["weighted avg"]["f1-score"] * 100, 2),
            "Architecture": "Ensemble (150 Decision Trees)",
        })
        plot_confusion_matrix(rf_cm, classes, "Random Forest", cm_dir / "rf_confusion_matrix.png")

    # Deep learning models
    dl_models = [
        ("Spatial CNN", RESULTS_DIR / "cnn_results.json", "cnn"),
        ("Temporal LSTM", RESULTS_DIR / "lstm_results.json", "lstm"),
        ("Hybrid CNN + LSTM", RESULTS_DIR / "cnn_lstm_results.json", "cnn_lstm"),
    ]

    from sklearn.metrics import classification_report

    for model_title, res_path, key in dl_models:
        if res_path.exists():
            with open(res_path, "r") as f:
                res = json.load(f)

            records.append({
                "Model": model_title,
                "Accuracy (%)": round(res["accuracy"] * 100, 2),
                "Macro Precision (%)": round(res["macro_precision"] * 100, 2),
                "Macro Recall (%)": round(res["macro_recall"] * 100, 2),
                "Macro F1 (%)": round(res["macro_f1"] * 100, 2),
                "Weighted F1 (%)": round(res["weighted_f1"] * 100, 2),
                "Architecture": model_title,
            })

            cm = np.array(res["confusion_matrix"])
            plot_confusion_matrix(cm, classes, model_title, cm_dir / f"{key}_confusion_matrix.png")

            if "history" in res:
                plot_training_curves(res["history"], model_title, curves_dir / f"{key}_training_curve.png")

            # Generate and export per-class classification report
            if "y_true" in res and "y_pred" in res:
                unique_labels = sorted(list(set(res["y_true"] + res["y_pred"])))
                target_names = [classes[idx] for idx in unique_labels]

                rep_dict = classification_report(
                    res["y_true"], res["y_pred"], labels=unique_labels, target_names=target_names, output_dict=True, zero_division=0
                )
                rep_txt = classification_report(
                    res["y_true"], res["y_pred"], labels=unique_labels, target_names=target_names, digits=4, zero_division=0
                )

                txt_path = reports_dir / f"{key}_report.txt"
                json_path = reports_dir / f"{key}_report.json"
                with open(txt_path, "w", encoding="utf-8") as tf:
                    tf.write(f"=== {model_title} Classification Report ===\n\n{rep_txt}")
                with open(json_path, "w", encoding="utf-8") as jf:
                    json.dump(rep_dict, jf, indent=2)
                logger.info(f"Generated per-class classification report: {txt_path}")

    # Generate master comparison table
    df = pd.DataFrame(records)
    csv_path = RESULTS_DIR / "metrics.csv"
    df.to_csv(csv_path, index=False)

    logger.info("\n" + "=" * 80)
    logger.info("FINAL UNIFIED HAR MODEL COMPARISON TABLE:")
    logger.info("=" * 80)
    logger.info("\n" + df.to_string(index=False))
    logger.info("=" * 80)
    logger.info(f"Saved master metrics to: {csv_path}")


if __name__ == "__main__":
    generate_unified_evaluation()
