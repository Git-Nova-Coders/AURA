"""
AURA Vision Module
Provides camera abstractions, pretrained YOLO object detection, multi-object tracking (M5),
visual annotations, structured feature extraction (FeatureBuilder), unified stream pipeline (VisionPipeline),
and dataset collection utilities (DatasetCollector).
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
from .features import DetectionFeatures, FeatureBuilder
from .tracker import ObjectTracker, TrackedObject, compute_iou
from .pipeline import VisionPipeline, PipelineResult
from .dataset_collector import DatasetCollector, LabeledFeatureSample

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
    "DetectionFeatures",
    "FeatureBuilder",
    "ObjectTracker",
    "TrackedObject",
    "compute_iou",
    "VisionPipeline",
    "PipelineResult",
    "DatasetCollector",
    "LabeledFeatureSample",
]
