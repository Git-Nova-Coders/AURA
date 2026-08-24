"""
AURA Configuration Module
Defines default settings and dataclass-based configurations for AURA subsystems.
"""

from dataclasses import dataclass, field
from typing import Union


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
    model_name: str = "yolo11n.pt"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = "auto"  # 'auto', 'cpu', 'cuda', etc.
    half: bool = False  # FP16 inference if CUDA is available


@dataclass
class FeaturesConfig:
    """Configuration for visual feature extraction (Feature Builder)."""
    enable_blur: bool = True
    enable_brightness: bool = True
    enable_contrast: bool = True
    normalize_quality: bool = True
    num_classes: int = 80


@dataclass
class DisplayConfig:
    """Configuration for visual rendering and UI display."""
    window_name: str = "AURA - Real-Time Visual Assistant"
    show_fps: bool = True
    show_labels: bool = True
    show_conf: bool = True
    show_features: bool = False
    box_thickness: int = 2
    font_scale: float = 0.6


@dataclass
class AuraConfig:
    """Master configuration for AURA application."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
