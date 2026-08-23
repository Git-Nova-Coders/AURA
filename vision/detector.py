"""
AURA Vision Detector Module
Provides pretrained YOLO-based object detection, structured detection outputs, and frame annotation.
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
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": [round(float(c), 2) for c in self.bbox],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Detection":
        """Creates a Detection instance from a dictionary."""
        return cls(
            class_id=int(data["class_id"]),
            class_name=str(data["class_name"]),
            confidence=float(data["confidence"]),
            bbox=[float(c) for c in data["bbox"]],
            track_id=int(data["track_id"]) if data.get("track_id") is not None else None,
        )


class DetectorError(Exception):
    """Base exception for detector-related errors."""
    pass


class ModelLoadError(DetectorError):
    """Raised when the detection model cannot be loaded."""
    pass


# Deterministic distinct colors for classes based on class_id
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
    
    Args:
        image: Original BGR image numpy array.
        detections: List of Detection objects.
        show_labels: Whether to render class names.
        show_conf: Whether to render confidence percentages.
        box_thickness: Thickness of bounding box borders.
        font_scale: Scale factor for label text.
        font_thickness: Thickness of label text.
        
    Returns:
        np.ndarray: Annotated BGR image.
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
        if show_labels:
            parts.append(det.class_name)
        if show_conf:
            parts.append(f"{det.confidence * 100:.1f}%")
        if det.track_id is not None:
            parts.insert(0, f"#{det.track_id}")

        if parts:
            label = " ".join(parts)
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            
            # Position badge above bbox, or inside if near top edge
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
    Vision Engine object detector wrapping pretrained Ultralytics YOLO models.
    Loads model once during initialization and processes frames into structured Detections.
    """

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "auto",
        half: bool = False,
    ):
        """
        Initializes the YOLO object detector.
        
        Args:
            model_name: Name or path of pretrained YOLO weights (e.g. 'yolo11n.pt', 'yolov8n.pt').
            confidence_threshold: Default minimum confidence score to retain detections.
            iou_threshold: NMS IoU threshold.
            device: Device target ('cpu', 'cuda', 'auto', etc.).
            half: Whether to use half-precision FP16.
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.half = half
        self._model: Optional[YOLO] = None
        self._load_model()

    def _load_model(self) -> None:
        """Loads the YOLO model instance once during initialization."""
        # Check if weights exist in models/ subfolder or as specified
        resolved_path = self.model_name
        if not os.path.isabs(resolved_path) and not os.path.exists(resolved_path):
            models_dir_path = os.path.join("models", resolved_path)
            if os.path.exists(models_dir_path):
                resolved_path = models_dir_path

        logger.info(f"Loading YOLO model '{resolved_path}' on device '{self.device}'...")
        try:
            self._model = YOLO(resolved_path)
            logger.info(f"YOLO model '{resolved_path}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model '{resolved_path}': {e}", exc_info=True)
            raise ModelLoadError(
                f"Could not load YOLO model '{resolved_path}'. Error: {e}"
            ) from e

    @property
    def class_names(self) -> Dict[int, str]:
        """Returns the dictionary mapping class IDs to class names."""
        if self._model is not None and hasattr(self._model, "names"):
            return self._model.names
        return {}

    def detect(
        self,
        frame: Union[np.ndarray, Frame],
        conf_threshold: Optional[float] = None,
        classes: Optional[List[int]] = None,
    ) -> List[Detection]:
        """
        Performs object detection on a single frame.
        
        Args:
            frame: Either an OpenCV BGR numpy array or an AURA Frame object.
            conf_threshold: Optional override for the detection confidence threshold.
            classes: Optional list of class IDs to filter for.
            
        Returns:
            List[Detection]: List of structured Detection objects.
        """
        if self._model is None:
            raise DetectorError("Detector model is not loaded.")

        img = frame.image if isinstance(frame, Frame) else frame
        if img is None or not isinstance(img, np.ndarray) or img.size == 0:
            logger.warning("Received empty or invalid image frame for detection.")
            return []

        conf = conf_threshold if conf_threshold is not None else self.confidence_threshold

        predict_kwargs: Dict[str, Any] = {
            "source": img,
            "conf": conf,
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
        xyxy = boxes.xyxy.cpu().numpy()  # [N, 4]
        confs = boxes.conf.cpu().numpy()  # [N]
        cls_ids = boxes.cls.cpu().numpy().astype(int)  # [N]

        for i in range(len(cls_ids)):
            c_id = int(cls_ids[i])
            c_name = self.class_names.get(c_id, str(c_id))
            c_conf = float(confs[i])
            c_bbox = [float(xyxy[i][0]), float(xyxy[i][1]), float(xyxy[i][2]), float(xyxy[i][3])]

            detections.append(
                Detection(
                    class_id=c_id,
                    class_name=c_name,
                    confidence=c_conf,
                    bbox=c_bbox,
                )
            )

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
