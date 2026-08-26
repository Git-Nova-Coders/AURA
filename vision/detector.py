"""
AURA Vision Detector Module
Supports standard YOLO models and YOLO-World Open-Vocabulary models with custom vocabularies,
structured detection outputs, per-class confidence thresholds, geometric anomaly suppression,
and frame annotation.
"""

import os
import logging
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

from .camera import Frame

logger = logging.getLogger(__name__)

# Default comprehensive indoor/assistant vocabulary for YOLO-World Open-Vocabulary detection
DEFAULT_AURA_VOCABULARY: List[str] = [
    "person",
    "laptop",
    "notebook",
    "book",
    "smartphone",
    "pen",
    "pencil",
    "water bottle",
    "cup",
    "backpack",
    "handbag",
    "headphones",
    "glasses",
    "keyboard",
    "computer mouse",
    "chair",
    "desk",
    "wrist watch",
]

# Class-specific confidence thresholds to suppress common hallucinations
DEFAULT_CLASS_THRESHOLDS: Dict[str, float] = {
    "laptop": 0.45,      # High accuracy now that 'notebook' is a distinct class
    "notebook": 0.35,    # Dedicated notebook detection
    "pen": 0.30,         # Fine-object detection
    "person": 0.40,      # Prevents floating hands/arms
    "smartphone": 0.40,  # Clear phone detection
    "headphones": 0.35,  # Audio gear
    "water bottle": 0.35,
    "cup": 0.35,
    "book": 0.38,
    "backpack": 0.40,
    "handbag": 0.40,
    "glasses": 0.30,
}


@dataclass
class Detection:
    """
    Structured representation of a single object detection.
    Independent of YOLO internals for clean inter-module communication.
    """
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2] in pixel coordinates
    track_id: Optional[int] = None
    reliability_score: Optional[float] = None
    reliability_label: Optional[str] = None

    @property
    def x1(self) -> float:
        return self.bbox[0]

    @property
    def y1(self) -> float:
        return self.bbox[1]

    @property
    def x2(self) -> float:
        return self.bbox[2]

    @property
    def y2(self) -> float:
        return self.bbox[3]

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def normalized_bbox(self, img_width: int, img_height: int) -> List[float]:
        """Returns [norm_x1, norm_y1, norm_x2, norm_y2] bounded in [0, 1]."""
        if img_width <= 0 or img_height <= 0:
            return [0.0, 0.0, 0.0, 0.0]
        return [
            max(0.0, min(1.0, self.x1 / img_width)),
            max(0.0, min(1.0, self.y1 / img_height)),
            max(0.0, min(1.0, self.x2 / img_width)),
            max(0.0, min(1.0, self.y2 / img_height)),
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Converts Detection to a standard dictionary matching the SSD specification."""
        d = {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": [round(float(c), 2) for c in self.bbox],
        }
        if self.reliability_score is not None:
            d["reliability_score"] = round(float(self.reliability_score), 4)
            d["reliability_label"] = self.reliability_label
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Detection":
        """Creates a Detection instance from a dictionary."""
        return cls(
            class_id=int(data["class_id"]),
            class_name=str(data["class_name"]),
            confidence=float(data["confidence"]),
            bbox=[float(c) for c in data["bbox"]],
            track_id=int(data["track_id"]) if data.get("track_id") is not None else None,
            reliability_score=float(data["reliability_score"]) if data.get("reliability_score") is not None else None,
            reliability_label=str(data["reliability_label"]) if data.get("reliability_label") is not None else None,
        )


class DetectorError(Exception):
    """Base exception for detector-related errors."""
    pass


class ModelLoadError(DetectorError):
    """Raised when the detection model cannot be loaded."""
    pass


def get_class_color(class_id: int) -> Tuple[int, int, int]:
    """Generates a consistent, visually distinct BGR color for each class ID."""
    np.random.seed(class_id * 37 + 101)
    color = np.random.randint(50, 255, size=3).tolist()
    return (int(color[0]), int(color[1]), int(color[2]))


def draw_detections(
    image: np.ndarray,
    detections: List[Detection],
    show_labels: bool = True,
    show_conf: bool = True,
    box_thickness: int = 2,
    font_scale: float = 0.5,
    font_thickness: int = 1,
) -> np.ndarray:
    """
    Renders structured detections on an image frame with clean bounding boxes and badges.
    """
    annotated = image.copy()
    img_h, img_w = annotated.shape[:2]

    for det in detections:
        x1 = int(max(0, min(img_w - 1, round(det.x1))))
        y1 = int(max(0, min(img_h - 1, round(det.y1))))
        x2 = int(max(0, min(img_w - 1, round(det.x2))))
        y2 = int(max(0, min(img_h - 1, round(det.y2))))

        color = get_class_color(det.class_id)

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness, lineType=cv2.LINE_AA)

        # Build label text
        parts = []
        if det.track_id is not None:
            parts.append(f"#{det.track_id}")
        if show_labels:
            parts.append(det.class_name)
        if show_conf:
            parts.append(f"{det.confidence * 100:.1f}%")
        if det.reliability_label is not None:
            parts.append(f"({det.reliability_label})")

        if parts:
            label = " ".join(parts)
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            
            badge_y1 = max(0, y1 - text_h - baseline - 4)
            badge_y2 = y1 if y1 - text_h - baseline - 4 >= 0 else y1 + text_h + baseline + 4
            badge_x2 = min(img_w - 1, x1 + text_w + 6)
            text_baseline_y = y1 - 4 if y1 - text_h - baseline - 4 >= 0 else y1 + text_h + 2

            # Background rectangle for text
            cv2.rectangle(
                annotated,
                (x1, badge_y1),
                (badge_x2, badge_y2),
                color,
                cv2.FILLED,
            )

            # High contrast text color (black or white)
            luminance = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
            text_color = (0, 0, 0) if luminance > 128 else (255, 255, 255)

            cv2.putText(
                annotated,
                label,
                (x1 + 3, text_baseline_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                font_thickness,
                lineType=cv2.LINE_AA,
            )

    return annotated


class ObjectDetector:
    """
    Vision Engine object detector wrapping Ultralytics YOLO & YOLO-World Open-Vocabulary models.
    Supports dynamic custom vocabulary, per-class thresholds, and geometric anomaly suppression.
    """

    def __init__(
        self,
        model_name: str = "yolov8m-worldv2.pt",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: str = "auto",
        half: bool = False,
        enable_geometric_filter: bool = True,
        class_thresholds: Optional[Dict[str, float]] = None,
        custom_classes: Optional[List[str]] = None,
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.half = half
        self.enable_geometric_filter = enable_geometric_filter
        self.class_thresholds = class_thresholds or DEFAULT_CLASS_THRESHOLDS
        self.custom_classes = custom_classes or (
            DEFAULT_AURA_VOCABULARY if "world" in model_name.lower() else None
        )
        self._model: Optional[YOLO] = None
        self._load_model()

    def _load_model(self) -> None:
        """Loads the YOLO model instance and configures custom vocabulary if supported."""
        resolved_path = self.model_name
        if not os.path.isabs(resolved_path) and not os.path.exists(resolved_path):
            models_dir_path = os.path.join("models", resolved_path)
            if os.path.exists(models_dir_path):
                resolved_path = models_dir_path

        logger.info(f"Loading YOLO model '{resolved_path}' on device '{self.device}'...")
        try:
            self._model = YOLO(resolved_path)
            # If YOLO-World model, apply custom vocabulary
            if self.custom_classes and hasattr(self._model, "set_classes"):
                self._model.set_classes(self.custom_classes)
                logger.info(f"YOLO-World configured with {len(self.custom_classes)} custom classes: {self.custom_classes}")
            logger.info(f"YOLO model '{resolved_path}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model '{resolved_path}': {e}", exc_info=True)
            raise ModelLoadError(
                f"Could not load YOLO model '{resolved_path}'. Error: {e}"
            ) from e

    def set_custom_classes(self, classes: List[str]) -> None:
        """Dynamically updates the Open-Vocabulary class list at runtime."""
        self.custom_classes = list(classes)
        if self._model is not None and hasattr(self._model, "set_classes"):
            self._model.set_classes(self.custom_classes)
            logger.info(f"Updated YOLO-World vocabulary ({len(self.custom_classes)} classes): {self.custom_classes}")

    @property
    def class_names(self) -> Dict[int, str]:
        """Returns the dictionary mapping class IDs to class names."""
        if self.custom_classes:
            return {i: name for i, name in enumerate(self.custom_classes)}
        if self._model is not None and hasattr(self._model, "names"):
            return self._model.names
        return {}

    def _filter_geometric_anomalies(
        self,
        detections: List[Detection],
        img_w: int,
        img_h: int,
    ) -> List[Detection]:
        """
        Suppresses common false positive classifications using spatial heuristics:
        1. Hands / Arms misclassified as 'person' (small area or horizontal ratio).
        2. Micro noise boxes (< 100 px).
        """
        frame_area = float(img_w * img_h)
        clean_detections: List[Detection] = []

        for d in detections:
            w = d.width
            h = d.height
            area = d.area
            norm_area = area / max(frame_area, 1.0)
            aspect_ratio = (w / h) if h > 1e-5 else 1.0

            # Filter 1: Discard micro-noise (< 100 pixels)
            if area < 100.0:
                continue

            # Filter 2: Person Anomaly Suppression (Rules out hands, wrists, skin patches)
            if d.class_name.lower() == "person":
                if norm_area < 0.02 and d.confidence < 0.88:
                    continue
                if aspect_ratio > 1.4 and d.confidence < 0.85:
                    continue

            clean_detections.append(d)

        return clean_detections

    def detect(
        self,
        frame: Union[np.ndarray, Frame],
        conf_threshold: Optional[float] = None,
        classes: Optional[List[int]] = None,
    ) -> List[Detection]:
        """
        Performs object detection on a single frame.
        """
        if self._model is None:
            raise DetectorError("Detector model is not loaded.")

        img = frame.image if isinstance(frame, Frame) else frame
        if img is None or not isinstance(img, np.ndarray) or img.size == 0:
            logger.warning("Received empty or invalid image frame for detection.")
            return []

        base_conf = conf_threshold if conf_threshold is not None else self.confidence_threshold

        predict_kwargs: Dict[str, Any] = {
            "source": img,
            "conf": max(0.12, base_conf - 0.15),  # Predict with lower floor then apply per-class thresholds
            "iou": self.iou_threshold,
            "verbose": False,
        }
        if self.device != "auto":
            predict_kwargs["device"] = self.device
        if self.half:
            predict_kwargs["half"] = True
        if classes is not None:
            predict_kwargs["classes"] = classes

        results = self._model.predict(**predict_kwargs)

        detections: List[Detection] = []
        if not results:
            return detections

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)

        current_class_names = self.class_names

        for i in range(len(cls_ids)):
            c_id = int(cls_ids[i])
            c_name = current_class_names.get(c_id, str(c_id))
            c_conf = float(confs[i])
            c_bbox = [float(xyxy[i][0]), float(xyxy[i][1]), float(xyxy[i][2]), float(xyxy[i][3])]

            # Check per-class threshold
            class_req_conf = self.class_thresholds.get(c_name.lower(), base_conf)
            if c_conf < class_req_conf:
                continue

            detections.append(
                Detection(
                    class_id=c_id,
                    class_name=c_name,
                    confidence=c_conf,
                    bbox=c_bbox,
                )
            )

        if self.enable_geometric_filter:
            img_h, img_w = img.shape[:2]
            detections = self._filter_geometric_anomalies(detections, img_w, img_h)

        return detections

    def annotate(
        self,
        frame: Union[np.ndarray, Frame],
        detections: List[Detection],
        show_labels: bool = True,
        show_conf: bool = True,
        box_thickness: int = 2,
    ) -> np.ndarray:
        """Helper method to annotate a frame with detections."""
        img = frame.image if isinstance(frame, Frame) else frame
        return draw_detections(
            image=img,
            detections=detections,
            show_labels=show_labels,
            show_conf=show_conf,
            box_thickness=box_thickness,
        )

    def __repr__(self) -> str:
        return (
            f"<ObjectDetector(model='{self.model_name}', conf_threshold={self.confidence_threshold}, "
            f"device='{self.device}')>"
        )
