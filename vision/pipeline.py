"""
AURA Vision Pipeline Module
Provides a high-level, decoupled pipeline and generator interface for downstream modules
(Reliability ANN, Interface UI, Knowledge Engine, Context Manager).
Integrates YOLO object detection, multi-object tracking (M5), and OCR (M5).
"""

import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Union, Generator, Dict, Any
import numpy as np

from config.config import AuraConfig
from .camera import CameraAdapter, Frame
from .detector import ObjectDetector, Detection
from .features import FeatureBuilder, DetectionFeatures
from .tracker import ObjectTracker
from ocr.engine import OCREngine, TextDetection, draw_text_annotations

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
    text_detections: List[TextDetection] = field(default_factory=list)
    object_texts: Dict[int, List[TextDetection]] = field(default_factory=dict)

    @property
    def num_detections(self) -> int:
        return len(self.detections)

    @property
    def num_texts(self) -> int:
        return len(self.text_detections)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the entire perception result into a JSON-compatible dictionary."""
        return {
            "timestamp": self.frame.timestamp,
            "source_id": self.frame.source_id,
            "frame_shape": list(self.frame.shape),
            "num_detections": self.num_detections,
            "num_texts": self.num_texts,
            "fps": round(self.fps, 1),
            "latency_ms": round(self.latency_ms, 2),
            "detections": [d.to_dict() for d in self.detections],
            "features": [f.to_dict() for f in self.features],
            "texts": [t.to_dict() for t in self.text_detections],
        }


class VisionPipeline:
    """
    High-level orchestrator combining CameraAdapter, ObjectDetector, ObjectTracker,
    FeatureBuilder, Reliability ANN, and OCREngine.
    """

    def __init__(
        self,
        config: Optional[AuraConfig] = None,
        camera: Optional[CameraAdapter] = None,
        detector: Optional[ObjectDetector] = None,
        feature_builder: Optional[FeatureBuilder] = None,
        tracker: Optional[ObjectTracker] = None,
        ocr_engine: Optional[OCREngine] = None,
    ):
        self.config = config or AuraConfig()

        # 1. Initialize detector
        self.detector = detector or ObjectDetector(
            model_name=self.config.vision.model_name,
            confidence_threshold=self.config.vision.confidence_threshold,
            iou_threshold=self.config.vision.iou_threshold,
            device=self.config.vision.device,
            half=self.config.vision.half,
        )

        # 2. Initialize tracker
        if tracker is not None:
            self.tracker = tracker
        elif self.config.tracker.enabled:
            self.tracker = ObjectTracker(
                max_age=self.config.tracker.max_age,
                min_hits=self.config.tracker.min_hits,
                iou_threshold=self.config.tracker.iou_threshold,
            )
        else:
            self.tracker = None

        # 3. Initialize OCR engine
        if ocr_engine is not None:
            self.ocr_engine = ocr_engine
        elif self.config.ocr.enabled:
            self.ocr_engine = OCREngine(
                languages=self.config.ocr.languages,
                confidence_threshold=self.config.ocr.confidence_threshold,
                gpu=self.config.ocr.gpu,
            )
        else:
            self.ocr_engine = None

        # 4. Initialize feature builder
        self.feature_builder = feature_builder or FeatureBuilder(
            enable_blur=self.config.features.enable_blur,
            enable_brightness=self.config.features.enable_brightness,
            enable_contrast=self.config.features.enable_contrast,
            enable_temporal=self.config.features.enable_temporal,
        )

        # 5. Initialize Reliability ANN
        from ann.inference import ReliabilityInference
        self.reliability_ann = ReliabilityInference(
            enabled=self.config.ann.enabled,
            model_path=self.config.ann.model_path,
            scaler_path=self.config.ann.scaler_path,
            confidence_threshold=self.config.ann.confidence_threshold,
            device=self.config.ann.device,
        )

        # 6. Camera adapter (lazy-opened if streaming)
        self.camera = camera or CameraAdapter(
            source=self.config.camera.source,
            width=self.config.camera.width,
            height=self.config.camera.height,
            fps=self.config.camera.fps,
        )

        self._last_timestamp = time.perf_counter()
        self._fps_ema = 0.0
        self._frame_index = 0
        self._last_text_detections: List[TextDetection] = []
        self._last_object_texts: Dict[int, List[TextDetection]] = {}

    def process_frame(
        self,
        frame: Union[np.ndarray, Frame],
        extract_features: bool = True,
        annotate: bool = True,
        run_ocr: Optional[bool] = None,
    ) -> PipelineResult:
        """
        Runs the full perception pipeline on a single frame.
        
        Args:
            frame: Numpy array or Frame object.
            extract_features: Whether to extract numerical features for detections.
            annotate: Whether to render bounding box annotations onto the frame.
            run_ocr: Explicit override to run OCR on this frame. If None, uses config stride.
            
        Returns:
            PipelineResult: Structured detections, features, text, and metadata.
        """
        t0 = time.perf_counter()
        self._frame_index += 1

        if isinstance(frame, np.ndarray):
            frame_obj = Frame(image=frame, timestamp=time.time(), source_id="direct_frame")
        else:
            frame_obj = frame

        # 1. Object Detection
        detections = self.detector.detect(frame_obj)

        # 2. Multi-Object Tracking (Persistent track IDs)
        tracks_map: Dict[int, Any] = {}
        if self.tracker is not None:
            detections = self.tracker.update(detections)
            tracks_map = {t.track_id: t for t in self.tracker.all_tracks}

        # 3. Feature Building & Reliability Estimation
        features: List[DetectionFeatures] = []
        if extract_features and detections:
            features = self.feature_builder.extract_all(frame_obj, detections, tracks_map=tracks_map)
            for i, feat in enumerate(features):
                rel_res = self.reliability_ann.predict(feat)
                detections[i].reliability_score = rel_res.score
                detections[i].reliability_label = rel_res.label

        # 4. OCR Extraction (Strided or on-demand)
        should_run_ocr = run_ocr if run_ocr is not None else (
            self.ocr_engine is not None and (self._frame_index % max(1, self.config.ocr.stride) == 0)
        )

        text_detections = self._last_text_detections
        object_texts = self._last_object_texts

        if should_run_ocr and self.ocr_engine is not None:
            text_detections = self.ocr_engine.extract_text(frame_obj)
            object_texts = self.ocr_engine.extract_text_for_detections(frame_obj.image, detections)
            self._last_text_detections = text_detections
            self._last_object_texts = object_texts

        # 5. Annotation Rendering
        annotated: Optional[np.ndarray] = None
        if annotate:
            annotated = self.detector.annotate(
                frame_obj,
                detections,
                show_labels=self.config.display.show_labels,
                show_conf=self.config.display.show_conf,
                box_thickness=self.config.display.box_thickness,
            )
            # Render OCR annotations if enabled and texts found
            if self.config.display.show_ocr and text_detections and annotated is not None:
                annotated = draw_text_annotations(annotated, text_detections)

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
            text_detections=text_detections,
            object_texts=object_texts,
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
