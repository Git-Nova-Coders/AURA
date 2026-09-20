import os
import json
import torch
import numpy as np

from .model import get_model, AnomalyCNN
from .preprocessing import preprocess_image
from . import config

class AnomalyDetector:
    """
    Inference Engine for Human Anomaly Detection.
    Provides a clean, reusable interface for external projects (like Aura).
    Automatically inspects model_config.json to load the appropriate architecture.
    """
    def __init__(self, model_path="models/anomaly_model.pth"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}.")
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Check if architecture is defined in model_config.json
        config_path = os.path.join(config.MODELS_DIR, "model_config.json")
        arch = "baseline"
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    meta = json.load(f)
                    arch = meta.get("architecture", "baseline")
            except Exception:
                arch = "baseline"

        # Initialize appropriate architecture
        try:
            self.model = get_model(arch, pretrained=False)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        except Exception:
            # Fallback to baseline if state_dict fails with selected arch
            self.model = AnomalyCNN()
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self, person_image: np.ndarray) -> dict:
        """
        Accepts a raw BGR image (as a NumPy array, e.g., from OpenCV or Aura camera frame),
        preprocesses it, and returns the anomaly classification and confidence score.
        """
        if person_image is None or not isinstance(person_image, np.ndarray) or person_image.size == 0:
            return {
                "error": "Invalid image input provided."
            }
            
        try:
            # 1. Apply EXACT same preprocessing used during training (is_training=False)
            processed_img = preprocess_image(person_image, is_training=False)
            
            # 2. Add batch dimension (H, W, C) -> (1, H, W, C)
            batch_img = np.expand_dims(processed_img, axis=0)
            
            # 3. Convert to PyTorch tensor and move to device
            input_tensor = torch.tensor(batch_img, dtype=torch.float32).to(self.device)
            
            # 4. Generate prediction
            with torch.no_grad():
                output = self.model(input_tensor)
                prob = torch.sigmoid(output).item()
                
            # 5. Convert prediction to label and confidence
            # Threshold is 0.5; Label 0 is Normal, Label 1 is Anomaly
            if prob >= 0.5:
                label = "Anomaly"
                confidence = prob
            else:
                label = "Normal"
                confidence = 1.0 - prob
                
            return {
                "label": label,
                "confidence": round(float(confidence), 4)
            }
            
        except Exception as e:
            return {
                "error": f"Prediction failed: {str(e)}"
            }
