"""
AURA Reliability ANN Inference Module
Loads trained model weights and preprocessing parameters to score object detections.
Implements a fallback mode to make YOLO-only decisions if the ANN is disabled or fails to load.
"""

import os
import pickle
import logging
from dataclasses import dataclass
from typing import List, Optional, Union
import numpy as np
import torch

from ann.model import ReliabilityANN
from vision.features import DetectionFeatures

logger = logging.getLogger("AURA.ANN.Inference")


@dataclass
class ReliabilityResult:
    """Structured response containing reliability predictions for downstream consumption."""
    score: float
    label: str  # "reliable" or "unreliable"
    model_version: str


class ReliabilityInference:
    """
    Inference wrapper for the Reliability ANN.
    Manages loading weights, scaler transformations, model execution, and fallback strategies.
    """

    def __init__(
        self,
        enabled: bool = True,
        model_path: str = "models/reliability_ann.pth",
        scaler_path: str = "models/scaler.pkl",
        confidence_threshold: float = 0.5,
        device: str = "cpu",
    ):
        self.enabled = enabled
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.confidence_threshold = confidence_threshold
        self.device_name = device
        
        # Internal state
        self.model: Optional[ReliabilityANN] = None
        self.scaler = None
        self.include_class_onehot = False
        self.feature_family = "all"
        self.input_dim = 14
        self.model_version = "fallback_yolo"
        
        # Load artifacts if enabled
        if self.enabled:
            self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Loads weights and scaler. Catches errors and triggers fallback mode if anything fails."""
        try:
            if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
                raise FileNotFoundError(
                    f"Model weights or scaler file missing. Model: '{self.model_path}', Scaler: '{self.scaler_path}'"
                )

            # 1. Load Scaler
            with open(self.scaler_path, "rb") as f:
                scaler_metadata = pickle.load(f)
            
            self.scaler = scaler_metadata["scaler"]
            self.include_class_onehot = scaler_metadata.get("include_class_onehot", False)
            self.feature_family = scaler_metadata.get("feature_family", "all")
            self.input_dim = scaler_metadata.get("input_dim", 14)
            self.model_version = scaler_metadata.get("model_version", "ann_v1")

            # 2. Determine hardware device
            if self.device_name == "auto" or self.device_name == "cuda":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(self.device_name)

            # 3. Load model weights
            self.model = ReliabilityANN(input_dim=self.input_dim)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(
                f"Successfully loaded Reliability ANN version '{self.model_version}' ({self.input_dim} features) on '{self.device}'"
            )

        except Exception as e:
            logger.warning(
                f"Reliability ANN initialization failed: {e}. Falling back to YOLO confidence mode."
            )
            self.enabled = False
            self.model = None
            self.scaler = None
            self.model_version = "fallback_yolo"

    def predict(self, features: DetectionFeatures) -> ReliabilityResult:
        """
        Runs reliability prediction for a single detection feature set.
        
        Args:
            features (DetectionFeatures): Extracted visual features.
            
        Returns:
            ReliabilityResult: Score, label, and model version.
        """
        if not self.enabled or self.model is None or self.scaler is None:
            # Fallback mode: Use YOLO confidence threshold directly
            score = float(features.confidence)
            label = "reliable" if score >= self.confidence_threshold else "unreliable"
            return ReliabilityResult(
                score=score,
                label=label,
                model_version=self.model_version,
            )

        try:
            # 1. Format feature vector
            raw_vector = features.to_vector(
                include_class_onehot=self.include_class_onehot,
                num_classes=80,  # default max classes
            )

            # Slice raw vector to match feature family if necessary
            if self.feature_family == "geometry":
                raw_vector = raw_vector[:11]
            elif self.feature_family == "quality":
                raw_vector = raw_vector[11:14]

            # 2. Reshape and Scale
            raw_2d = raw_vector.reshape(1, -1)
            scaled_vector = self.scaler.transform(raw_2d)

            # 3. PyTorch Inference
            x_tensor = torch.tensor(scaled_vector, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                prob = self.model(x_tensor).cpu().item()

            label = "reliable" if prob >= self.confidence_threshold else "unreliable"
            return ReliabilityResult(
                score=float(prob),
                label=label,
                model_version=self.model_version,
            )

        except Exception as e:
            logger.error(f"ANN prediction error: {e}. Returning fallback YOLO confidence prediction.")
            score = float(features.confidence)
            label = "reliable" if score >= self.confidence_threshold else "unreliable"
            return ReliabilityResult(
                score=score,
                label=label,
                model_version="fallback_yolo_error",
            )

    def predict_batch(self, features_list: List[DetectionFeatures]) -> List[ReliabilityResult]:
        """Runs batch inference on a list of detection features."""
        return [self.predict(f) for f in features_list]
