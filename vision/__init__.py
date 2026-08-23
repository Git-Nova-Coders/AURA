"""
AURA Vision Module
Provides camera abstractions, pretrained YOLO object detection, and visual annotations.
"""

from .camera import Frame, CameraAdapter, CameraError, CameraNotFoundError
from .detector import (
    Detection,
    ObjectDetector,
    DetectorError,
    ModelLoadError,
    draw_detections,
    get_class_color,
)

__all__ = [
    "Frame",
    "CameraAdapter",
    "CameraError",
    "CameraNotFoundError",
    "Detection",
    "ObjectDetector",
    "DetectorError",
    "ModelLoadError",
    "draw_detections",
    "get_class_color",
]
