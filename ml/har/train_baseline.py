"""
AURA HAR Random Forest Baseline Model
Trains a Random Forest classifier on kinematic and statistical pose features.
Evaluates strictly on official UCF101 test split.
Reports: Accuracy, Macro/Weighted Precision, Recall, F1, Confusion Matrix.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.har.features import extract_features_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("AURA.HAR.Baseline")

DATA_DIR = PROJECT_ROOT / "data" / "har" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "har"
RESULTS_DIR = PROJECT_ROOT / "results" / "har"


def train_baseline_rf(
    n_estimators: int = 150,
    max_depth: int = 15,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Trains and evaluates Random Forest baseline.
    """
    logger.info("=" * 60)
    logger.info("PHASE 3: TRAINING RANDOM FOREST BASELINE")
    logger.info("=" * 60)

    # 1. Load preprocessed sequence datasets
    X_train_seq = np.load(DATA_DIR / "X_train.npy")  # (N_train, 30, 33, 4)
    y_train = np.load(DATA_DIR / "y_train.npy")      # (N_train,)
    X_test_seq = np.load(DATA_DIR / "X_test.npy")    # (N_test, 30, 33, 4)
    y_test = np.load(DATA_DIR / "y_test.npy")        # (N_test,)

    with open(DATA_DIR / "dataset_meta.json", "r") as f:
        meta = json.load(f)
    classes = meta["classes"]

    logger.info(f"Loaded train sequences: {X_train_seq.shape}, test sequences: {X_test_seq.shape}")

    # 2. Extract handcrafted kinematic & statistical features
    logger.info("Extracting handcrafted kinematic features for training set...")
    X_train_feat = extract_features_dataset(X_train_seq)
    logger.info("Extracting handcrafted kinematic features for test set...")
    X_test_feat = extract_features_dataset(X_test_seq)
    logger.info(f"Feature matrix shape: Train {X_train_feat.shape}, Test {X_test_feat.shape}")

    # 3. Train Random Forest Classifier
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    logger.info("Fitting Random Forest on training set...")
    rf.fit(X_train_feat, y_train)

    # 4. Evaluate on Official UCF101 Test Set
    y_pred = rf.predict(X_test_feat)

    acc = float(accuracy_score(y_test, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)

    report_str = classification_report(y_test, y_pred, target_names=classes, digits=4, zero_division=0)
    report_dict = classification_report(y_test, y_pred, target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    logger.info("\n" + "=" * 60)
    logger.info("RANDOM FOREST BASELINE EVALUATION (OFFICIAL TEST SPLIT):")
    logger.info("=" * 60)
    logger.info(f"Accuracy:       {acc * 100:.2f}%")
    logger.info(f"Macro Prec:     {p_macro * 100:.2f}%")
    logger.info(f"Macro Recall:   {r_macro * 100:.2f}%")
    logger.info(f"Macro F1-score: {f1_macro * 100:.2f}%")
    logger.info(f"Weighted F1:    {f1_weight * 100:.2f}%")
    logger.info("\nClassification Report:\n" + report_str)
    logger.info("\nConfusion Matrix:\n" + str(cm))

    # 5. Save model and results
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "classification_reports").mkdir(exist_ok=True)
    (RESULTS_DIR / "confusion_matrices").mkdir(exist_ok=True)

    model_path = MODEL_DIR / "random_forest_baseline.joblib"
    joblib.dump(rf, model_path)
    logger.info(f"Saved model to: {model_path}")

    metrics = {
        "model": "Random Forest Baseline",
        "accuracy": acc,
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weight),
        "hyperparameters": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state,
        },
        "num_train": len(y_train),
        "num_test": len(y_test),
        "num_features": X_train_feat.shape[1],
    }

    with open(RESULTS_DIR / "classification_reports" / "random_forest_report.json", "w") as f:
        json.dump(report_dict, f, indent=2)

    np.save(RESULTS_DIR / "confusion_matrices" / "random_forest_cm.npy", cm)

    return metrics


if __name__ == "__main__":
    train_baseline_rf()
