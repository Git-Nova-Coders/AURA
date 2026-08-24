"""
AURA Vision Pipeline Module
Provides a high-level, decoupled pipeline and generator interface for downstream modules
(Reliability ANN, Interface UI, Knowledge Engine, Context Manager).
"""

import time
import logging
from dataclasses import dataclass
from typing import List, Optional, Union, Generator, Dict, Any
import numpy as np

from config.config import AuraConfig
from .camera import CameraAdapter, Frame
from .detector import ObjectDetector, Detection
from .features import FeatureBuilder, DetectionFeatures

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """
    Consolidated output produced by the vision pipeline for a single frame.
    Enables other modules to consume perception output without low-level model details.
    """
    frame: Frame
    detections: List[Detection]
    features: List[DetectionFeatures]
    annotated_frame: Optional[np.ndarray] = None
    fps: float = 0.0
    latency_ms: float = 0.0

    @property
    def num_detections(self) -> int:
        return len(self.detections)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the entire perception result into a JSON-compatible dictionary."""
        return {
            "timestamp": self.frame.timestamp,
            "source_id": self.frame.source_id,
            "frame_shape": list(self.frame.shape),
            "num_detections": self.num_detections,
            "fps": round(self.fps, 1),
            "latency_ms": round(self.latency_ms, 2),
            "detections": [d.to_dict() for d in self.detections],
            "features": [f.to_dict() for f in self.features],
        }


class VisionPipeline:
    """
    High-level orchestrator combining CameraAdapter, ObjectDetector, and FeatureBuilder.
    Provides single-frame processing and real-time streaming generators.
    """

    def __init__(
        self,
        config: Optional[AuraConfig] = None,
        camera: Optional[CameraAdapter] = None,
        detector: Optional[ObjectDetector] = None,
        feature_builder: Optional[FeatureBuilder] = None,
    ):
        self.config = config or AuraConfig()

        # Initialize detector
        self.detector = detector or ObjectDetector(
            model_name=self.config.vision.model_name,
            confidence_threshold=self.config.vision.confidence_threshold,
            iou_threshold=self.config.vision.iou_threshold,
            device=self.config.vision.device,
            half=self.config.vision.half,
        )

        # Initialize feature builder
        self.feature_builder = feature_builder or FeatureBuilder(
            enable_blur=self.config.features.enable_blur,
            enable_brightness=self.config.features.enable_brightness,
            enable_contrast=self.config.features.enable_contrast,
        )

        # Initialize Reliability ANN
        from ann.inference import ReliabilityInference
        self.reliability_ann = ReliabilityInference(
            enabled=self.config.ann.enabled,
            model_path=self.config.ann.model_path,
            scaler_path=self.config.ann.scaler_path,
            confidence_threshold=self.config.ann.confidence_threshold,
            device=self.config.ann.device,
        )

        # Camera adapter (lazy-opened if streaming)
        self.camera = camera or CameraAdapter(
            source=self.config.camera.source,
            width=self.config.camera.width,
            height=self.config.camera.height,
            fps=self.config.camera.fps,
        )

        self._last_timestamp = time.perf_counter()
        self._fps_ema = 0.0

    def process_frame(
        self,
        frame: Union[np.ndarray, Frame],
        extract_features: bool = True,
        annotate: bool = True,
    ) -> PipelineResult:
        """
        Runs the full perception pipeline on a single frame.
        
        Args:
            frame: Numpy array or Frame object.
            extract_features: Whether to extract numerical features for detections.
            annotate: Whether to render bounding box annotations onto the frame.
            
        Returns:
            PipelineResult: Structured detections, features, and metadata.
        """
        t0 = time.perf_counter()

        if isinstance(frame, np.ndarray):
            frame_obj = Frame(image=frame, timestamp=time.time(), source_id="direct_frame")
        else:
            frame_obj = frame

        # 1. Object Detection
        detections = self.detector.detect(frame_obj)
        t_detect = time.perf_counter()

        # 2. Feature Building & Reliability Estimation
        features: List[DetectionFeatures] = []
        if extract_features and detections:
            features = self.feature_builder.extract_all(frame_obj, detections)
            for i, feat in enumerate(features):
                rel_res = self.reliability_ann.predict(feat)
                detections[i].reliability_score = rel_res.score
                detections[i].reliability_label = rel_res.label

        # 3. Annotation
        annotated: Optional[np.ndarray] = None
        if annotate:
            annotated = self.detector.annotate(
                frame_obj,
                detections,
                show_labels=self.config.display.show_labels,
                show_conf=self.config.display.show_conf,
                box_thickness=self.config.display.box_thickness,
            )

        t_end = time.perf_counter()
        latency_ms = (t_end - t0) * 1000.0

        # FPS calculation
        frame_interval = t_end - self._last_timestamp
        self._last_timestamp = t_end
        current_fps = 1.0 / max(frame_interval, 1e-6)
        self._fps_ema = current_fps if self._fps_ema == 0.0 else (0.1 * current_fps + 0.9 * self._fps_ema)

        return PipelineResult(
            frame=frame_obj,
            detections=detections,
            features=features,
            annotated_frame=annotated,
            fps=self._fps_ema,
            latency_ms=latency_ms,
        )

    def stream(
        self,
        extract_features: bool = True,
        annotate: bool = True,
        max_frames: Optional[int] = None,
    ) -> Generator[PipelineResult, None, None]:
        """
        Generates PipelineResults continuously from the camera stream.
        
        Yields:
            PipelineResult: Result for each incoming frame.
        """
        opened_here = False
        if not self.camera.is_opened:
            self.camera.open()
            opened_here = True

        frame_count = 0
        try:
            while True:
                frame = self.camera.read()
                if frame is None:
                    break

                result = self.process_frame(
                    frame=frame,
                    extract_features=extract_features,
                    annotate=annotate,
                )
                yield result

                frame_count += 1
                if max_frames is not None and frame_count >= max_frames:
                    break
        finally:
            if opened_here:
                self.camera.release()

    def open(self) -> "VisionPipeline":
        if not self.camera.is_opened:
            self.camera.open()
        return self

    def release(self) -> None:
        if self.camera.is_opened:
            self.camera.release()

    def __enter__(self) -> "VisionPipeline":
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
