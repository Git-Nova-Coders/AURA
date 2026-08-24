"""
AURA Feature Builder Module
Extracts geometric, confidence, and image-quality features from object detections.
Produces standardized numerical feature vectors for the downstream Reliability ANN (Milestone 3).
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
import cv2
import numpy as np

from .camera import Frame
from .detector import Detection

logger = logging.getLogger(__name__)


@dataclass
class DetectionFeatures:
    """
    Standardized numerical features extracted from a single Detection and its image crop.
    Decoupled from YOLO internals to serve as training/inference input for the Reliability ANN.
    """
    # Core identifiers & prediction
    class_id: int
    class_name: str
    confidence: float

    # Normalized Bounding Box Geometry [0.0, 1.0]
    norm_x1: float
    norm_y1: float
    norm_x2: float
    norm_y2: float
    norm_width: float
    norm_height: float
    norm_area: float
    norm_center_x: float
    norm_center_y: float
    aspect_ratio: float  # width / height

    # Visual Quality Metrics (computed from bbox crop)
    blur_score: float     # Laplacian variance (higher = sharper, lower = blurrier)
    brightness: float     # Mean pixel luminance in range [0.0, 255.0]
    contrast: float       # Standard deviation of pixel intensities in range [0.0, 128.0]

    # Optional Tracking metadata (for future temporal persistence in M5)
    track_id: Optional[int] = None

    def to_vector(
        self,
        include_class_onehot: bool = False,
        num_classes: int = 80,
    ) -> np.ndarray:
        """
        Converts the feature dataclass into a 1D NumPy array for PyTorch / ANN inference.
        
        Vector order:
        [confidence, norm_x1, norm_y1, norm_x2, norm_y2, norm_width, norm_height,
         norm_area, norm_center_x, norm_center_y, aspect_ratio, blur_score, brightness, contrast]
        + optional one-hot class encoding [c_0, ..., c_{num_classes-1}]
        
        Returns:
            np.ndarray: 1D float32 array of features.
        """
        base_features = [
            self.confidence,
            self.norm_x1,
            self.norm_y1,
            self.norm_x2,
            self.norm_y2,
            self.norm_width,
            self.norm_height,
            self.norm_area,
            self.norm_center_x,
            self.norm_center_y,
            self.aspect_ratio,
            self.blur_score,
            self.brightness,
            self.contrast,
        ]

        if include_class_onehot:
            one_hot = [0.0] * num_classes
            if 0 <= self.class_id < num_classes:
                one_hot[self.class_id] = 1.0
            base_features.extend(one_hot)

        return np.array(base_features, dtype=np.float32)

    @classmethod
    def feature_names(
        cls,
        include_class_onehot: bool = False,
        num_classes: int = 80,
    ) -> List[str]:
        """Returns standard column names for the numerical feature vector."""
        names = [
            "confidence",
            "norm_x1",
            "norm_y1",
            "norm_x2",
            "norm_y2",
            "norm_width",
            "norm_height",
            "norm_area",
            "norm_center_x",
            "norm_center_y",
            "aspect_ratio",
            "blur_score",
            "brightness",
            "contrast",
        ]
        if include_class_onehot:
            names.extend([f"class_{i}" for i in range(num_classes)])
        return names

    def to_dict(self) -> Dict[str, Any]:
        """Serializes features to a JSON-compatible dictionary."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "norm_geometry": {
                "x1": round(float(self.norm_x1), 4),
                "y1": round(float(self.norm_y1), 4),
                "x2": round(float(self.norm_x2), 4),
                "y2": round(float(self.norm_y2), 4),
                "width": round(float(self.norm_width), 4),
                "height": round(float(self.norm_height), 4),
                "area": round(float(self.norm_area), 4),
                "center_x": round(float(self.norm_center_x), 4),
                "center_y": round(float(self.norm_center_y), 4),
                "aspect_ratio": round(float(self.aspect_ratio), 4),
            },
            "visual_quality": {
                "blur_score": round(float(self.blur_score), 2),
                "brightness": round(float(self.brightness), 2),
                "contrast": round(float(self.contrast), 2),
            },
            "track_id": self.track_id,
        }


class FeatureBuilder:
    """
    Constructs standardized DetectionFeatures from raw frames and Detection objects.
    Extracts geometric, blur, brightness, and contrast indicators.
    """

    def __init__(
        self,
        enable_blur: bool = True,
        enable_brightness: bool = True,
        enable_contrast: bool = True,
    ):
        self.enable_blur = enable_blur
        self.enable_brightness = enable_brightness
        self.enable_contrast = enable_contrast

    def extract(
        self,
        frame: Union[np.ndarray, Frame],
        detection: Detection,
    ) -> DetectionFeatures:
        """
        Extracts numerical features from a single detection.
        
        Args:
            frame: OpenCV BGR image array or AURA Frame.
            detection: Structured Detection object.
            
        Returns:
            DetectionFeatures: Extracted feature representation.
        """
        img = frame.image if isinstance(frame, Frame) else frame
        img_h, img_w = img.shape[:2]

        # 1. Geometry Normalization [0.0, 1.0]
        if img_w > 0 and img_h > 0:
            norm_x1 = max(0.0, min(1.0, float(detection.x1) / img_w))
            norm_y1 = max(0.0, min(1.0, float(detection.y1) / img_h))
            norm_x2 = max(0.0, min(1.0, float(detection.x2) / img_w))
            norm_y2 = max(0.0, min(1.0, float(detection.y2) / img_h))
        else:
            norm_x1, norm_y1, norm_x2, norm_y2 = 0.0, 0.0, 0.0, 0.0

        norm_w = max(0.0, norm_x2 - norm_x1)
        norm_h = max(0.0, norm_y2 - norm_y1)
        norm_area = norm_w * norm_h
        norm_cx = (norm_x1 + norm_x2) / 2.0
        norm_cy = (norm_y1 + norm_y2) / 2.0
        aspect_ratio = (norm_w / norm_h) if norm_h > 1e-6 else 0.0

        # 2. Crop Object Region for Visual Quality Analysis
        x1 = max(0, min(img_w - 1, int(round(detection.x1))))
        y1 = max(0, min(img_h - 1, int(round(detection.y1))))
        x2 = max(x1 + 1, min(img_w, int(round(detection.x2))))
        y2 = max(y1 + 1, min(img_h, int(round(detection.y2))))

        crop = img[y1:y2, x1:x2]

        blur_score = 0.0
        brightness = 0.0
        contrast = 0.0

        if crop.size > 0:
            # Convert to grayscale for luminance & sharpness computation
            if len(crop.shape) == 3 and crop.shape[2] == 3:
                gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            else:
                gray_crop = crop

            # Visual Quality 1: Sharpness (Laplacian Variance)
            if self.enable_blur and gray_crop.shape[0] >= 2 and gray_crop.shape[1] >= 2:
                laplacian = cv2.Laplacian(gray_crop, cv2.CV_64F)
                blur_score = float(laplacian.var())

            # Visual Quality 2: Brightness (Mean Intensity)
            if self.enable_brightness:
                brightness = float(np.mean(gray_crop))

            # Visual Quality 3: Contrast (Standard Deviation)
            if self.enable_contrast:
                contrast = float(np.std(gray_crop))

        return DetectionFeatures(
            class_id=detection.class_id,
            class_name=detection.class_name,
            confidence=float(detection.confidence),
            norm_x1=norm_x1,
            norm_y1=norm_y1,
            norm_x2=norm_x2,
            norm_y2=norm_y2,
            norm_width=norm_w,
            norm_height=norm_h,
            norm_area=norm_area,
            norm_center_x=norm_cx,
            norm_center_y=norm_cy,
            aspect_ratio=aspect_ratio,
            blur_score=blur_score,
            brightness=brightness,
            contrast=contrast,
            track_id=detection.track_id,
        )

    def extract_all(
        self,
        frame: Union[np.ndarray, Frame],
        detections: List[Detection],
    ) -> List[DetectionFeatures]:
        """
        Extracts features for all detections in a frame.
        
        Args:
            frame: OpenCV BGR image array or AURA Frame.
            detections: List of Detection objects.
            
        Returns:
            List[DetectionFeatures]: Extracted features for each detection.
        """
        return [self.extract(frame, det) for det in detections]
