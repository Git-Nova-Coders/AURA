"""
AURA HAR Phase 5: Temporal Pose LSTM Training
Trains a Bidirectional LSTM Network on Temporal Pose Sequences.
Saves model checkpoint, training history, and evaluation metrics.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.har.models import TemporalPoseLSTM
from ml.har.train_utils import prepare_dataloaders, train_model, get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("AURA.HAR.TrainLSTM")

DATA_DIR = PROJECT_ROOT / "data" / "har" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "har"
RESULTS_DIR = PROJECT_ROOT / "results" / "har"


def run_lstm_training(epochs: int = 60, batch_size: int = 32, lr: float = 1e-3):
    logger.info("=" * 60)
    logger.info("PHASE 5: TRAINING TEMPORAL POSE LSTM")
    logger.info("=" * 60)

    train_loader, val_loader, test_loader, split_info = prepare_dataloaders(
        DATA_DIR, batch_size=batch_size
    )

    device = get_device()
    model = TemporalPoseLSTM(
        in_features=132,
        hidden_size=128,
        num_layers=2,
        num_classes=8,
        dropout=0.3,
        bidirectional=True
    )
    model_save_path = MODEL_DIR / "lstm_har_model.pt"

    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model_save_path=model_save_path,
        model_name="Temporal LSTM",
        epochs=epochs,
        lr=lr,
        device=device
    )

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "lstm_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Phase 5 LSTM Training completed and metrics saved.")
    return results


if __name__ == "__main__":
    run_lstm_training()
