"""
AURA Pipeline Bridge (Milestone 9)
Thread-safe bridge between the AURA vision/reasoning pipeline and the FastAPI web server.
Maintains double-buffered state for zero-contention reads from WebSocket/REST handlers.
"""

import time
import base64
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

import cv2
import numpy as np

from config.config import (
    SAHIConfig, RAGConfig, MemoryConfig, IntelligenceConfig,
)
from vision.camera import CameraAdapter, CameraNotFoundError, Frame
from vision.detector import ObjectDetector, ModelLoadError, Detection
from vision.tracker import ObjectTracker
from vision.features import FeatureBuilder
from ocr.engine import OCREngine, TextDetection, draw_text_annotations
from ann.inference import ReliabilityInference
from knowledge.retriever import KnowledgeRetriever
from knowledge.rag import RAGEngine
from brain.context import ContextManager, SceneContext, ObjectEntity
from brain.conversation import ConversationEngine, ConversationResponse
from brain.memory import EpisodicMemory
from brain.llm import create_llm_provider
from vision.gestures import (
    GestureActionController,
    GestureMode,
    GestureResult,
    GestureType,
    draw_hand_skeleton,
    draw_action_toast,
)

logger = logging.getLogger("AURA.Bridge")


@dataclass
class TelemetrySnapshot:
    """Thread-safe telemetry state snapshot."""
    fps: float = 0.0
    inference_latency_ms: float = 0.0
    frame_count: int = 0
    active_tracks: int = 0
    detection_count: int = 0
    ocr_text_count: int = 0
    sahi_enabled: bool = False
    tracking_enabled: bool = True
    ocr_enabled: bool = True
    voice_listening: bool = False
    ann_version: Optional[str] = None
    voice_status: str = "OFF"
    memory_enabled: bool = False
    rag_enabled: bool = False
    gesture_mode: str = "ALL_OBJECTS"
    active_gesture: str = "none"
    pointed_target: Optional[str] = None
    pointed_target_bbox: Optional[List[float]] = None
    active_toast: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fps": round(self.fps, 1),
            "inference_latency_ms": round(self.inference_latency_ms, 1),
            "frame_count": self.frame_count,
            "active_tracks": self.active_tracks,
            "detection_count": self.detection_count,
            "ocr_text_count": self.ocr_text_count,
            "sahi_enabled": self.sahi_enabled,
            "tracking_enabled": self.tracking_enabled,
            "ocr_enabled": self.ocr_enabled,
            "voice_listening": self.voice_listening,
            "ann_version": self.ann_version,
            "voice_status": self.voice_status,
            "memory_enabled": self.memory_enabled,
            "rag_enabled": self.rag_enabled,
            "gesture_mode": self.gesture_mode,
            "active_gesture": self.active_gesture,
            "pointed_target": self.pointed_target,
            "pointed_target_bbox": self.pointed_target_bbox,
            "active_toast": self.active_toast,
            "timestamp": round(self.timestamp, 2),
        }


class AuraBridge:
    """
    Thread-safe bridge connecting the AURA vision pipeline to the web interface.
    Maintains buffered state for concurrent reads from multiple WebSocket clients.
    """

    def __init__(
        self,
        source: str = "synthetic",
        model_name: str = "yolov8m-worldv2.pt",
        conf_thresh: float = 0.35,
        device: str = "auto",
        width: int = 640,
        height: int = 480,
        custom_classes: Optional[str] = None,
        enable_sahi: bool = False,
        slice_size: int = 320,
        slice_overlap: float = 0.20,
        enable_ocr: bool = True,
        ocr_stride: int = 30,
        no_ann: bool = False,
        ann_model: str = "models/reliability_ann.pth",
        ann_scaler: str = "models/scaler.pkl",
        ann_thresh: float = 0.25,
        enable_rag: bool = True,
        rag_dir: str = "data/manuals",
        enable_memory: bool = True,
        memory_db: str = "data/memory.db",
        llm_provider: str = "offline",
    ):
        self.source = source
        self.width = width
        self.height = height

        # Thread-safe state buffers
        self._lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._running = False

        # Buffered state
        self._latest_annotated_jpeg: Optional[bytes] = None
        self._latest_scene: Optional[SceneContext] = None
        self._latest_detections: List[Detection] = []
        self._telemetry = TelemetrySnapshot()

        # Parse custom classes
        classes_list = (
            [c.strip() for c in custom_classes.split(",") if c.strip()]
            if custom_classes else None
        )

        # SAHI config
        sahi_config = SAHIConfig(
            enabled=enable_sahi,
            slice_width=slice_size,
            slice_height=slice_size,
            overlap_width_ratio=slice_overlap,
            overlap_height_ratio=slice_overlap,
        )

        # Initialize pipeline components
        logger.info("Initializing AURA Pipeline Bridge...")

        try:
            self.detector = ObjectDetector(
                model_name=model_name,
                confidence_threshold=conf_thresh,
                device=device,
                custom_classes=classes_list,
                sahi_config=sahi_config,
            )
        except ModelLoadError as e:
            logger.error(f"Detector init failed: {e}")
            raise

        self.tracker = ObjectTracker(
            max_age=30, min_hits=1, iou_threshold=0.3, smooth_factor=0.65
        )
        self.feature_builder = FeatureBuilder()

        # OCR
        self.ocr_engine: Optional[OCREngine] = None
        self.ocr_stride = ocr_stride
        self.enable_ocr = enable_ocr
        if enable_ocr:
            try:
                self.ocr_engine = OCREngine(confidence_threshold=0.3)
            except Exception as e:
                logger.warning(f"OCR init failed: {e}. Running without OCR.")

        # ANN
        self.reliability_ann = ReliabilityInference(
            enabled=not no_ann,
            model_path=ann_model,
            scaler_path=ann_scaler,
            confidence_threshold=ann_thresh,
            device=device,
        )

        # Context Manager & Brain
        self.context_manager = ContextManager()
        self.knowledge_retriever = KnowledgeRetriever(
            enable_curated=True, enable_wikipedia=True,
        )

        # RAG
        self.rag_engine = RAGEngine(
            RAGConfig(enabled=enable_rag, docs_directory=rag_dir)
        )
        if enable_rag:
            self.rag_engine.initialize()

        # Episodic Memory
        self.episodic_memory = EpisodicMemory(
            MemoryConfig(enabled=enable_memory, db_path=memory_db)
        )

        # LLM & Conversation
        intel_cfg = IntelligenceConfig(llm_provider=llm_provider)
        llm_reasoner = create_llm_provider(intel_cfg)

        self.conversation_engine = ConversationEngine(
            context_manager=self.context_manager,
            knowledge_retriever=self.knowledge_retriever,
            rag_engine=self.rag_engine,
            memory=self.episodic_memory,
            llm_provider=llm_reasoner,
        )

        # Tracking state
        self._tracking_enabled = True
        self._ocr_enabled = enable_ocr
        self._voice_listening = False
        self._enable_memory = enable_memory
        self._enable_rag = enable_rag

        # Camera / synthetic
        self._camera: Optional[CameraAdapter] = None
        self._use_synthetic = False
        self._pipeline_thread: Optional[threading.Thread] = None

        # OCR cache
        self._last_ocr_texts: List[TextDetection] = []

        # Event broadcaster for pushing async events (e.g. gesture pinch inspect) to WebSockets
        self._event_broadcaster = None

        # Gesture Controller with real action callbacks
        self.gesture_controller = GestureActionController(
            on_inspect_callback=self.inspect_target,
            on_toggle_sahi_callback=self.toggle_sahi,
            on_voice_trigger_callback=self.toggle_voice,
        )

        logger.info("AURA Pipeline Bridge initialized successfully.")

    def _init_camera(self) -> None:
        """Initializes camera or synthetic frame source."""
        if self.source.lower() == "synthetic":
            self._use_synthetic = True
            logger.info("Using synthetic frame generator for dashboard.")
            return

        source_target = int(self.source) if self.source.isdigit() else self.source
        try:
            self._camera = CameraAdapter(
                source=source_target, width=self.width, height=self.height,
            )
            self._camera.open()
            logger.info(f"Camera connected: '{source_target}'")
        except CameraNotFoundError:
            logger.warning("Camera not found. Falling back to synthetic frames.")
            self._use_synthetic = True

    def _create_synthetic_frame(self, idx: int) -> Tuple[Frame, List[Detection], List[TextDetection]]:
        """
        Generates a rich, futuristic visual animation simulating object detection
        when the physical camera is off, unavailable, or in simulation mode.
        """
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # 1. Futuristic dark cyber background with subtle grid
        for y in range(0, self.height, 40):
            cv2.line(img, (0, y), (self.width, y), (28, 22, 18), 1)
        for x in range(0, self.width, 40):
            cv2.line(img, (x, 0), (x, self.height), (28, 22, 18), 1)

        # 2. Moving animated scanline (green-cyan glow)
        scan_y = int((idx * 4) % self.height)
        cv2.line(img, (0, scan_y), (self.width, scan_y), (120, 240, 200), 1)
        if scan_y > 2:
            cv2.line(img, (0, scan_y - 2), (self.width, scan_y - 2), (40, 100, 70), 1)

        # 3. Simulated Moving Person (Silhouette with futuristic aesthetic)
        cx = int(180 + 90 * np.sin(idx * 0.04))
        cy = int(220 + 8 * np.cos(idx * 0.08))
        # Torso & body
        cv2.rectangle(img, (cx - 32, cy - 70), (cx + 32, cy + 90), (45, 160, 85), -1)
        cv2.rectangle(img, (cx - 32, cy - 70), (cx + 32, cy + 90), (0, 240, 150), 2)
        # Head
        cv2.circle(img, (cx, cy - 95), 24, (45, 160, 85), -1)
        cv2.circle(img, (cx, cy - 95), 24, (0, 240, 150), 2)
        # Visor
        cv2.rectangle(img, (cx - 14, cy - 100), (cx + 14, cy - 90), (255, 240, 0), -1)
        # Legs
        cv2.line(img, (cx - 18, cy + 90), (cx - 22, cy + 150), (45, 160, 85), 8)
        cv2.line(img, (cx + 18, cy + 90), (cx + 22, cy + 150), (45, 160, 85), 8)

        # 4. Simulated Workstation Desk & Laptop
        desk_y = 360
        cv2.line(img, (240, desk_y), (620, desk_y), (80, 80, 80), 2)

        # Laptop screen & keyboard base
        lx, ly = 370, 310
        cv2.rectangle(img, (lx - 55, ly - 45), (lx + 55, ly + 25), (40, 35, 30), -1)
        cv2.rectangle(img, (lx - 55, ly - 45), (lx + 55, ly + 25), (240, 200, 0), 2)
        # Glowing laptop screen
        cv2.rectangle(img, (lx - 48, ly - 40), (lx + 48, ly + 18), (140, 90, 20), -1)
        cv2.putText(img, "AURA OS", (lx - 35, ly - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        # Laptop base
        pts_base = np.array([[lx - 65, ly + 40], [lx + 65, ly + 40], [lx + 55, ly + 25], [lx - 55, ly + 25]], np.int32)
        cv2.fillPoly(img, [pts_base], (60, 55, 50))
        cv2.polylines(img, [pts_base], True, (240, 200, 0), 1)

        # 5. Simulated Notebook with dynamic position
        nx = int(510 + 20 * np.cos(idx * 0.02))
        ny = 325
        cv2.rectangle(img, (nx - 38, ny - 28), (nx + 38, ny + 32), (140, 60, 30), -1)
        cv2.rectangle(img, (nx - 38, ny - 28), (nx + 38, ny + 32), (220, 120, 0), 2)
        cv2.putText(img, "MANUAL", (nx - 30, ny + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

        # 6. Water Bottle / Cup
        bx, by = 275, 330
        cv2.rectangle(img, (bx - 12, by - 25), (bx + 12, by + 25), (160, 110, 40), -1)
        cv2.rectangle(img, (bx - 12, by - 25), (bx + 12, by + 25), (255, 180, 0), 1)
        cv2.rectangle(img, (bx - 6, by - 33), (bx + 6, by - 25), (100, 80, 80), -1)

        # 7. Simulation HUD watermark
        cv2.putText(img, "CAMERA OFF // SIMULATION ENGINE ACTIVE", (16, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 240, 255), 1)

        synth_frame = Frame(
            image=img,
            timestamp=time.time(),
            source_id="synthetic",
        )

        synth_dets = [
            Detection(class_id=0, class_name="person", confidence=0.95, bbox=[float(cx - 45), float(cy - 125), float(cx + 45), float(cy + 155)]),
            Detection(class_id=1, class_name="laptop", confidence=0.91, bbox=[float(lx - 68), float(ly - 50), float(lx + 68), float(ly + 45)]),
            Detection(class_id=2, class_name="notebook", confidence=0.88, bbox=[float(nx - 45), float(ny - 35), float(nx + 45), float(ny + 40)]),
            Detection(class_id=3, class_name="bottle", confidence=0.84, bbox=[float(bx - 15), float(by - 35), float(bx + 15), float(by + 30)]),
            Detection(class_id=4, class_name="face", confidence=0.90, bbox=[float(cx - 20), float(cy - 110), float(cx + 20), float(cy - 80)]),
        ]

        synth_texts = [
            TextDetection(text="AURA OS", confidence=0.96, bbox=[lx - 35, ly - 20, lx + 35, ly + 5]),
            TextDetection(text="MANUAL", confidence=0.94, bbox=[nx - 30, ny - 5, nx + 30, ny + 15]),
        ]

        return synth_frame, synth_dets, synth_texts

    def _pipeline_loop(self) -> None:
        """Main vision pipeline loop running in a background thread."""
        self._init_camera()
        frame_count = 0
        fps_ema = 0.0
        alpha = 0.1

        logger.info("Pipeline loop started.")

        # Decoupled async inference worker for live camera / video sources
        current_camera_dets: List[Detection] = []
        det_lock = threading.Lock()
        latest_camera_frame: Optional[Frame] = None
        frame_lock = threading.Lock()
        last_infer_latency = 0.0

        def async_inference_worker():
            nonlocal current_camera_dets, last_infer_latency
            while self._running:
                work_frame = None
                with frame_lock:
                    if latest_camera_frame is not None:
                        work_frame = latest_camera_frame
                if work_frame is not None:
                    t0_inf = time.perf_counter()
                    dets = self.detector.detect(work_frame)
                    t1_inf = time.perf_counter()
                    with det_lock:
                        current_camera_dets = dets
                        last_infer_latency = (t1_inf - t0_inf) * 1000.0
                time.sleep(0.005)

        ocr_lock = threading.Lock()
        def async_ocr_worker():
            while self._running:
                if self.ocr_engine is not None:
                    work_frame = None
                    with frame_lock:
                        if latest_camera_frame is not None:
                            work_frame = latest_camera_frame
                    if work_frame is not None:
                        try:
                            texts = self.ocr_engine.extract_text(work_frame)
                            with ocr_lock:
                                self._last_ocr_texts = texts
                        except Exception as e:
                            logger.debug(f"Async OCR extract error: {e}")
                # Scan OCR in background at 1.5s interval without blocking video stream
                time.sleep(1.5)

        if not self._use_synthetic:
            infer_thread = threading.Thread(target=async_inference_worker, daemon=True, name="AURA_Async_Infer")
            infer_thread.start()
            if self.ocr_engine is not None:
                ocr_thread = threading.Thread(target=async_ocr_worker, daemon=True, name="AURA_Async_OCR")
                ocr_thread.start()

        while self._running:
            t0 = time.perf_counter()

            # 1. Capture frame & detect
            if self._use_synthetic:
                frame, detections, synth_texts = self._create_synthetic_frame(frame_count)
                self._last_ocr_texts = synth_texts
                infer_latency_ms = 15.0
            else:
                try:
                    frame = self._camera.read()
                except Exception as e:
                    logger.warning(f"Camera read error: {e}. Falling back to simulation.")
                    frame = None

                if frame is None:
                    # Seamless animation fallback when camera is off or unavailable
                    frame, detections, synth_texts = self._create_synthetic_frame(frame_count)
                    self._last_ocr_texts = synth_texts
                    infer_latency_ms = 15.0
                else:
                    with frame_lock:
                        latest_camera_frame = frame
                    with det_lock:
                        detections = list(current_camera_dets)
                    infer_latency_ms = last_infer_latency

            # 3. Track
            tracks_map: Dict[int, Any] = {}
            if self._tracking_enabled:
                detections = self.tracker.update(detections)
                tracks_map = {t.track_id: t for t in self.tracker.all_tracks}

            # 4. Feature extraction & ANN reliability
            features = (
                self.feature_builder.extract_all(
                    frame, detections, tracks_map=tracks_map,
                )
                if detections else []
            )
            if features:
                for i, feat in enumerate(features):
                    rel_res = self.reliability_ann.predict(feat)
                    detections[i].reliability_score = rel_res.score
                    detections[i].reliability_label = rel_res.label

            # 5. Non-blocking OCR state sync
            with ocr_lock:
                current_ocr_texts = list(self._last_ocr_texts)

            # 6. Context update
            object_texts_map: Dict[int, List[TextDetection]] = {}
            if current_ocr_texts and detections:
                for idx, det in enumerate(detections):
                    key = det.track_id if det.track_id is not None else idx
                    dx1, dy1, dx2, dy2 = det.bbox
                    matching = []
                    for t in current_ocr_texts:
                        tcx, tcy = t.center
                        if dx1 <= tcx <= dx2 and dy1 <= tcy <= dy2:
                            matching.append(t)
                    if matching:
                        object_texts_map[key] = matching

            scene_context = self.context_manager.update(
                detections=detections,
                text_detections=current_ocr_texts,
                object_texts=object_texts_map,
                frame_shape=frame.shape,
            )

            # 7. Record memory
            if self._enable_memory and self.episodic_memory:
                self.episodic_memory.record_scene(scene_context)

            # 8. Hand Gesture Interactive Control
            raw_img = frame.image if isinstance(frame, Frame) else frame
            (
                visible_detections,
                gesture_mode,
                active_gesture,
                pointed_target,
            ) = self.gesture_controller.update(raw_img, detections)

            # 9. Annotate frame based on active gesture mode
            if gesture_mode == GestureMode.HIDE_BOXES:
                # Clean View: No bounding boxes rendered
                annotated = raw_img.copy()
                cv2.putText(
                    annotated,
                    "GESTURE MODE: 🖐️ HIDE ALL BOXES (OPEN PALM)",
                    (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 240, 255),
                    2,
                )
            else:
                annotated = self.detector.annotate(raw_img, visible_detections)
                if self._last_ocr_texts:
                    annotated = draw_text_annotations(annotated, self._last_ocr_texts)

                # Pointing Laser & Target Visual Indicator
                if (
                    gesture_mode in (GestureMode.FOCUS_OBJECT, GestureMode.INSPECT_OBJECT)
                    and pointed_target is not None
                    and active_gesture.pointing_tip is not None
                ):
                    ptx, pty = int(active_gesture.pointing_tip[0]), int(active_gesture.pointing_tip[1])
                    tx1, ty1, tx2, ty2 = [int(c) for c in pointed_target.bbox]
                    tcx, tcy = (tx1 + tx2) // 2, (ty1 + ty2) // 2
                    
                    # Draw laser ray with glow
                    cv2.line(annotated, (ptx, pty), (tcx, tcy), (0, 240, 255), 2, cv2.LINE_AA)
                    cv2.circle(annotated, (ptx, pty), 7, (0, 240, 255), -1, cv2.LINE_AA)
                    
                    # Target Crosshair Reticle
                    cv2.circle(annotated, (tcx, tcy), 14, (0, 255, 120), 2, cv2.LINE_AA)
                    cv2.drawMarker(annotated, (tcx, tcy), (0, 255, 120), cv2.MARKER_CROSS, 20, 2)
                    
                    tag_prefix = "👌 INSPECTING" if gesture_mode == GestureMode.INSPECT_OBJECT else "👉 LOCKED"
                    cv2.putText(
                        annotated,
                        f"{tag_prefix}: {pointed_target.class_name.upper()}",
                        (tx1, max(20, ty1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.60,
                        (0, 240, 255),
                        2,
                    )

            # Draw Cybernetic 21-Landmark Hand Skeleton
            if active_gesture.landmarks:
                annotated = draw_hand_skeleton(annotated, active_gesture)

            # Draw Real-Time HUD Action Toast Banner
            if self.gesture_controller.active_toast and time.time() < self.gesture_controller.toast_expiry_time:
                annotated = draw_action_toast(annotated, self.gesture_controller.active_toast)

            # 10. Encode to JPEG
            _, jpeg_buf = cv2.imencode(
                ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75],
            )

            # 11. FPS & Telemetry
            loop_time = time.perf_counter() - t0
            current_fps = 1.0 / max(loop_time, 1e-6)
            fps_ema = (
                current_fps if frame_count == 0
                else (alpha * current_fps + (1 - alpha) * fps_ema)
            )

            active_tracks = (
                len(self.tracker.active_tracks) if self._tracking_enabled else 0
            )
            ocr_count = len(self._last_ocr_texts)
            sahi_active = bool(
                self.detector.sahi_config and self.detector.sahi_config.enabled
            )

            # 12. Update buffered state (thread-safe)
            with self._lock:
                self._latest_annotated_jpeg = jpeg_buf.tobytes()
                self._latest_scene = scene_context
                self._latest_detections = list(visible_detections)
                
                toast_str = (
                    self.gesture_controller.active_toast
                    if (self.gesture_controller.active_toast and time.time() < self.gesture_controller.toast_expiry_time)
                    else None
                )

                target_bbox = pointed_target.bbox if pointed_target else None

                self._telemetry = TelemetrySnapshot(
                    fps=fps_ema,
                    inference_latency_ms=infer_latency_ms,
                    frame_count=frame_count,
                    active_tracks=active_tracks,
                    detection_count=len(visible_detections),
                    ocr_text_count=ocr_count,
                    sahi_enabled=sahi_active,
                    tracking_enabled=self._tracking_enabled,
                    ocr_enabled=self._ocr_enabled,
                    voice_listening=self._voice_listening,
                    ann_version=self.reliability_ann.model_version,
                    memory_enabled=self._enable_memory,
                    rag_enabled=self._enable_rag,
                    gesture_mode=gesture_mode.value,
                    active_gesture=active_gesture.gesture.value,
                    pointed_target=pointed_target.class_name if pointed_target else None,
                    pointed_target_bbox=target_bbox,
                    active_toast=toast_str,
                )

            frame_count += 1

            # Target ~30 FPS preview loop
            elapsed = time.perf_counter() - t0
            target_interval = 1.0 / 30.0
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)

        logger.info("Pipeline loop stopped.")

    # ── Public API (thread-safe) ──

    def start(self) -> None:
        """Starts the pipeline in a background thread."""
        if self._running:
            logger.warning("Pipeline already running.")
            return
        self._running = True
        self._pipeline_thread = threading.Thread(
            target=self._pipeline_loop, daemon=True, name="AURA_Pipeline",
        )
        self._pipeline_thread.start()
        logger.info("Pipeline background thread started.")

    def stop(self) -> None:
        """Stops the pipeline."""
        self._running = False
        if self._pipeline_thread:
            self._pipeline_thread.join(timeout=5.0)
        if self._camera:
            self._camera.release()
        if self._enable_memory and self.episodic_memory:
            self.episodic_memory.close()
        logger.info("Pipeline stopped and cleaned up.")

    @property
    def is_running(self) -> bool:
        return self._running

    def get_frame_jpeg(self) -> Optional[bytes]:
        """Returns the latest annotated frame as JPEG bytes."""
        with self._lock:
            return self._latest_annotated_jpeg

    def get_frame_base64(self) -> Optional[str]:
        """Returns the latest annotated frame as base64-encoded JPEG string."""
        jpeg = self.get_frame_jpeg()
        if jpeg:
            return base64.b64encode(jpeg).decode("ascii")
        return None

    def get_scene(self) -> Optional[Dict[str, Any]]:
        """Returns the latest scene context as a serializable dict."""
        with self._lock:
            scene = self._latest_scene
        if not scene:
            return None
        return {
            "timestamp": round(scene.timestamp, 2),
            "frame_index": scene.frame_index,
            "entity_count": scene.num_entities,
            "entities": [e.to_dict() for e in scene.entities],
            "texts": [t.to_dict() for t in scene.all_texts],
            "relations": [
                {"sentence": r.to_sentence()} for r in scene.spatial_relations
            ],
            "summary": scene.summary(),
            "frame_shape": list(scene.frame_shape),
        }

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns a telemetry snapshot dict."""
        with self._lock:
            return self._telemetry.to_dict()

    def get_detections(self) -> List[Dict[str, Any]]:
        """Returns the latest detections as a list of dicts."""
        with self._lock:
            return [d.to_dict() for d in self._latest_detections]

    def send_chat(self, query: str) -> Dict[str, Any]:
        """Processes a chat query through the conversation engine."""
        response = self.conversation_engine.respond(query)
        return response.to_dict()

    def search_rag(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Searches RAG documents."""
        result = self.rag_engine.query(query, top_k=top_k)
        return result.to_dict()

    def search_memory(self, query: str) -> Dict[str, Any]:
        """Searches episodic memory for an object."""
        event = self.episodic_memory.find_last_seen(query)
        if event:
            return {
                "found": True,
                "description": event.describe(),
                "event": event.to_dict(),
            }
        return {
            "found": False,
            "description": f"No memory record found for '{query}'.",
            "event": None,
        }

    def get_memory_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Returns recent episodic memory events."""
        if not self._enable_memory:
            return []
        # Query common object classes
        all_events = []
        for cls in ["face", "person", "laptop", "notebook", "phone", "cup", "bottle", "book"]:
            events = self.episodic_memory.get_history(cls, limit=5)
            all_events.extend(events)
        # Sort by timestamp descending, dedupe by id
        seen_ids = set()
        unique = []
        for e in sorted(all_events, key=lambda x: x.timestamp, reverse=True):
            if e.id not in seen_ids:
                seen_ids.add(e.id)
                unique.append(e.to_dict())
        return unique[:limit]

    def get_rag_documents(self) -> List[Dict[str, Any]]:
        """Returns list of all indexed RAG documents."""
        if not self.rag_engine.is_available:
            return []
        docs = list(self.rag_engine.vector_store.documents.values())
        return [d.to_dict() for d in docs]

    def toggle_sahi(self) -> bool:
        """Toggles SAHI on/off and returns new state."""
        if self.detector.sahi_config and self.detector.sahi_config.enabled:
            self.detector.disable_sahi()
            self.gesture_controller.trigger_toast("SAHI DISABLED", duration=1.5)
            return False
        else:
            self.detector.enable_sahi()
            self.gesture_controller.trigger_toast("🤘 SAHI HIGH-RES ENABLED (320px Slices)", duration=2.0)
            return True

    def toggle_tracking(self) -> bool:
        """Toggles tracking on/off and returns new state."""
        self._tracking_enabled = not self._tracking_enabled
        state_str = "ENABLED" if self._tracking_enabled else "DISABLED"
        self.gesture_controller.trigger_toast(f"TRACKING {state_str}", duration=1.5)
        return self._tracking_enabled

    def toggle_ocr(self) -> bool:
        """Toggles OCR on/off and returns new state."""
        self._ocr_enabled = not self._ocr_enabled
        state_str = "ENABLED" if self._ocr_enabled else "DISABLED"
        self.gesture_controller.trigger_toast(f"OCR TEXT SCANNER {state_str}", duration=1.5)
        return self._ocr_enabled

    def toggle_voice(self) -> bool:
        """Toggles Voice Assistant listening mode."""
        self._voice_listening = not self._voice_listening
        state_str = "LISTENING..." if self._voice_listening else "PAUSED"
        self.gesture_controller.trigger_toast(f"🤙 VOICE ASSISTANT {state_str}", duration=1.8)
        return self._voice_listening

    def set_event_broadcaster(self, broadcaster: Any) -> None:
        """Sets callback function to broadcast real-time events to connected WebSocket clients."""
        self._event_broadcaster = broadcaster

    def inspect_target(self, query_override: Optional[Any] = None) -> Dict[str, Any]:
        """Inspects the currently locked or targeted object with Multimodal AI reasoning."""
        with self._lock:
            target_name = self._telemetry.pointed_target if self._telemetry else None
            scene = self._latest_scene

        # 1. Resolve target entity metadata
        target_str = "object"
        confidence = 0.95
        spatial_pos = "CENTER"
        ocr_texts: List[str] = []

        if hasattr(query_override, 'class_name'):
            target_str = str(query_override.class_name)
            confidence = getattr(query_override, 'confidence', 0.95)
            ocr_texts = getattr(query_override, 'ocr_texts', [])
            if hasattr(query_override, 'bbox') and query_override.bbox:
                bx = (query_override.bbox[0] + query_override.bbox[2]) / 2.0
                by = (query_override.bbox[1] + query_override.bbox[3]) / 2.0
                spatial_pos = f"{'LEFT' if bx < 240 else 'RIGHT' if bx > 400 else 'CENTER'} {'TOP' if by < 180 else 'BOTTOM' if by > 300 else 'MID'}"
        elif isinstance(query_override, str) and query_override.strip():
            target_str = query_override.strip()
        elif target_name:
            target_str = target_name

        # 2. Retrieve grounded encyclopedic knowledge
        k_item = self.knowledge_retriever.retrieve(target_str)
        k_title = k_item.title if k_item else target_str.capitalize()
        k_category = k_item.category if k_item else "Visual Entity"
        k_summary = k_item.summary if k_item else f"A standard {target_str} observed in the visual field."
        k_source = k_item.source if k_item else "Curated Intelligence"

        # 3. Query Conversation Engine
        chat_query = f"What is this {target_str}?"
        chat_resp = self.send_chat(chat_query)
        response_text = chat_resp.get("response_text", "")

        # Fallback to rich structured breakdown if response is too brief or contains generic fallback text
        if not response_text or len(response_text) < 15 or "do not see" in response_text.lower() or "aura" in response_text.lower():
            response_text = f"{k_title} ({k_category}): {k_summary}"
            if ocr_texts:
                response_text += f" Extracted OCR text: {', '.join([f'\"{t}\"' for t in ocr_texts])}."
            response_text += f" (Spatial Region: {spatial_pos}, Confidence: {int(confidence * 100)}%)."

        self.gesture_controller.trigger_toast(f"👌 INSPECTED: {target_str.upper()}", duration=2.5)

        result = {
            "target": target_str,
            "response": {
                "query": f"Inspect {target_str}",
                "intent": "object_info",
                "response_text": response_text,
                "confidence": confidence,
                "spatial_pos": spatial_pos,
                "ocr_texts": ocr_texts,
                "sources": [k_source, "context_manager"],
                "timestamp": time.time(),
            },
            "timestamp": time.time(),
        }

        # Broadcast inspect and chat response to all connected WebSockets immediately!
        if self._event_broadcaster:
            try:
                self._event_broadcaster({
                    "type": "inspect_response",
                    "data": result,
                })
                self._event_broadcaster({
                    "type": "chat_response",
                    "data": result["response"],
                })
            except Exception as e:
                logger.warning(f"Failed to broadcast inspect response: {e}")

        return result

    def get_status(self) -> Dict[str, Any]:
        """Returns overall system status."""
        return {
            "version": "0.9.5",
            "pipeline_running": self._running,
            "source": self.source,
            "tracking_enabled": self._tracking_enabled,
            "sahi_enabled": bool(
                self.detector.sahi_config and self.detector.sahi_config.enabled
            ),
            "ocr_enabled": self._ocr_enabled,
            "voice_listening": self._voice_listening,
            "rag_enabled": self._enable_rag,
            "memory_enabled": self._enable_memory,
            "ann_version": self.reliability_ann.model_version,
            "rag_document_count": self.rag_engine.vector_store.count,
        }
