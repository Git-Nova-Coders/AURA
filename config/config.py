"""
AURA Configuration Module
Defines default settings and dataclass-based configurations for AURA subsystems.
"""

from dataclasses import dataclass, field
from typing import Union, List, Optional


@dataclass
class CameraConfig:
    """Configuration for camera adapter."""
    source: Union[int, str] = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    auto_reconnect: bool = True
    timeout_seconds: float = 5.0


@dataclass
class VisionConfig:
    """Configuration for object detection vision engine."""
    model_name: str = "yolov8m-worldv2.pt"  # YOLO-World v2 Medium: High-precision Open-Vocabulary detector
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    device: str = "auto"  # 'auto', 'cpu', 'cuda', etc.
    half: bool = False  # FP16 inference if CUDA is available
    enable_geometric_filter: bool = True  # Suppress false positives (hands as person)
    custom_classes: Optional[List[str]] = None  # Dynamic custom vocabulary (defaults to AURA vocabulary)


@dataclass
class TrackerConfig:
    """Configuration for multi-object tracking (Milestone 5)."""
    enabled: bool = True
    algorithm: str = "iou"  # 'iou' or 'bytetrack'
    max_age: int = 30  # Maximum frames to keep lost tracks alive
    min_hits: int = 2   # Minimum consecutive detections before confirming a track (filters 1-frame glitches)
    iou_threshold: float = 0.3  # IoU association threshold


@dataclass
class OCRConfig:
    """Configuration for Optical Character Recognition (Milestone 5)."""
    enabled: bool = False  # Off by default in high-FPS stream; can be toggled on or run on-demand
    languages: List[str] = field(default_factory=lambda: ["en"])
    confidence_threshold: float = 0.3
    stride: int = 15  # Run full-frame OCR every N frames to avoid blocking video stream
    gpu: bool = False  # Run EasyOCR on GPU if available


@dataclass
class FeaturesConfig:
    """Configuration for visual feature extraction (Feature Builder)."""
    enable_blur: bool = True
    enable_brightness: bool = True
    enable_contrast: bool = True
    enable_temporal: bool = True
    normalize_quality: bool = True
    num_classes: int = 80


@dataclass
class DisplayConfig:
    """Configuration for visual rendering and UI display."""
    window_name: str = "AURA - Real-Time Visual Assistant"
    show_fps: bool = True
    show_labels: bool = True
    show_conf: bool = True
    show_tracks: bool = True
    show_ocr: bool = True
    show_features: bool = False
    box_thickness: int = 2
    font_scale: float = 0.6


@dataclass
class AnnConfig:
    """Configuration for Reliability ANN."""
    enabled: bool = True
    model_path: str = "models/reliability_ann.pth"
    scaler_path: str = "models/scaler.pkl"
    confidence_threshold: float = 0.5  # decision threshold for reliable vs unreliable
    device: str = "cpu"  # 'cpu', 'cuda', 'auto'
    model_version: str = "ann_v1"


@dataclass
class AuraConfig:
    """Master configuration for AURA application."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    ann: AnnConfig = field(default_factory=AnnConfig)
