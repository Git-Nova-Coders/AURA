"""
AURA - Adaptive Understanding and Reasoning Architecture
Milestone 8: Multimodal Visual Agent with RAG, Vector Memory & Pluggable LLM Reasoning
(Camera -> YOLO/SAHI Detection -> Multi-Object Tracking -> OCR -> Reliability ANN -> Context Manager -> Episodic Memory -> RAG Engine -> Conversational LLM Reasoning -> Voice STT/TTS)

Entry point for running live detection, tracking, OCR, SAHI sliced inference, feature extraction,
episodic spatial memory, RAG document retrieval, natural-language visual reasoning, and voice interaction.
"""

import sys
import time
import argparse
import logging
import threading
from typing import Optional, List, Dict, Any
import cv2
import numpy as np

from config.config import SAHIConfig, RAGConfig, MemoryConfig, IntelligenceConfig
from vision.camera import CameraAdapter, CameraNotFoundError, Frame
from vision.detector import ObjectDetector, ModelLoadError, Detection
from vision.tracker import ObjectTracker
from vision.features import FeatureBuilder
from vision.dataset_collector import DatasetCollector
from ocr.engine import OCREngine, TextDetection, draw_text_annotations
from ann.inference import ReliabilityInference
from knowledge.retriever import KnowledgeRetriever
from knowledge.sources import KnowledgeItem
from knowledge.rag import RAGEngine
from brain.context import ContextManager, SceneContext
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
from voice.engine import VoiceAssistant
from voice.tts import TextToSpeech
from voice.stt import SpeechToText

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AURA.Main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AURA Visual Intelligence Assistant - Vision, Tracking, SAHI, Memory, RAG & Voice Assistant"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Camera device index (e.g., '0') or path to video/image file or 'synthetic'. Default: '0'",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8m-worldv2.pt",
        help="YOLO model path. Default: 'yolov8m-worldv2.pt'",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Detection confidence threshold in range [0.0, 1.0]. Default: 0.35",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Comma-separated custom class names for YOLO-World (e.g. 'person,laptop,notebook,pen,smartphone').",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Inference device ('auto', 'cpu', 'cuda', etc.). Default: 'auto'",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Requested capture width. Default: 640",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Requested capture height. Default: 480",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying GUI window.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run in benchmark mode without requiring webcam (uses synthetic frame generator).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Max frames to process before exiting automatically.",
    )
    parser.add_argument(
        "--collect-dataset",
        type=str,
        default=None,
        metavar="CSV_PATH",
        help="Save extracted feature vectors to specified CSV file path for ANN training dataset.",
    )
    parser.add_argument(
        "--no-track",
        action="store_true",
        help="Disable Multi-Object Tracking subsystem.",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Enable Optical Character Recognition (OCR) extraction.",
    )
    parser.add_argument(
        "--ocr-stride",
        type=int,
        default=30,
        help="Perform background OCR scan every N frames. Default: 30",
    )
    parser.add_argument(
        "--no-ann",
        action="store_true",
        help="Disable the Reliability ANN module (force YOLO-only fallback mode).",
    )
    parser.add_argument(
        "--ann-model",
        type=str,
        default="models/reliability_ann.pth",
        help="Path to trained Reliability ANN model weights (.pth). Default: 'models/reliability_ann.pth'",
    )
    parser.add_argument(
        "--ann-scaler",
        type=str,
        default="models/scaler.pkl",
        help="Path to fitted scaler metadata (.pkl). Default: 'models/scaler.pkl'",
    )
    parser.add_argument(
        "--ann-thresh",
        type=float,
        default=0.25,
        help="Decision threshold for reliable/unreliable label. Default: 0.25",
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Enable Voice Assistant (Speech-To-Text & Text-To-Speech).",
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable Text-To-Speech audio playback.",
    )
    parser.add_argument(
        "--rag",
        action="store_true",
        help="Enable Retrieval-Augmented Generation (RAG) on documentation & manuals.",
    )
    parser.add_argument(
        "--rag-dir",
        type=str,
        default="data/manuals",
        help="Directory containing manuals/guides for RAG indexing. Default: 'data/manuals'",
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        help="Enable persistent episodic & spatial memory.",
    )
    parser.add_argument(
        "--memory-db",
        type=str,
        default="data/memory.db",
        help="Path to SQLite episodic memory database file. Default: 'data/memory.db'",
    )
    parser.add_argument(
        "--llm",
        type=str,
        default="offline",
        help="LLM Reasoning provider ('offline', 'gemini', 'ollama', 'openai'). Default: 'offline'",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run a one-shot natural-language visual query upon initialization.",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Enable interactive chat prompt loop in console alongside visual assistant.",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Force synchronous single-threaded execution.",
    )
    parser.add_argument(
        "--sahi",
        action="store_true",
        help="Enable Slicing Aided Hyper Inference (SAHI) for fine-grained small object detection.",
    )
    parser.add_argument(
        "--slice-size",
        type=int,
        default=320,
        help="SAHI slice window dimension (square tile size in pixels). Default: 320",
    )
    parser.add_argument(
        "--slice-overlap",
        type=float,
        default=0.20,
        help="SAHI slice horizontal/vertical overlap ratio in range [0.0, 0.8]. Default: 0.20",
    )
    parser.add_argument(
        "--gestures",
        action="store_true",
        default=True,
        help="Enable interactive hand gesture tracking (Open Palm to clear, Pointing to focus). Default: True",
    )
    parser.add_argument(
        "--no-gestures",
        action="store_false",
        dest="gestures",
        help="Disable hand gesture interactive controls.",
    )
    return parser.parse_args()


def create_synthetic_frame(frame_idx: int = 0) -> Frame:
    """Generates a synthetic test frame with moving shapes and visible text for benchmarking."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Gradient background
    for y in range(480):
        img[y, :, 0] = int(20 + 40 * (y / 480))
        img[y, :, 1] = int(20 + 20 * (y / 480))
        img[y, :, 2] = int(40 + 60 * (y / 480))

    # Moving person-like rectangle
    cx = int(200 + 100 * np.sin(frame_idx * 0.05))
    cv2.rectangle(img, (cx - 40, 100), (cx + 40, 340), (60, 180, 75), -1)
    cv2.circle(img, (cx, 80), 30, (60, 180, 75), -1)

    # Moving stationary object (notebook)
    nx = int(420 + 50 * np.cos(frame_idx * 0.03))
    cv2.rectangle(img, (nx - 60, 260), (nx + 60, 380), (180, 100, 40), -1)
    cv2.putText(img, "AURA ARCHITECTURE GUIDE", (nx - 55, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

    return Frame(image=img, timestamp=time.time(), frame_id=frame_idx, source_id="synthetic")


def draw_hud(
    image: np.ndarray,
    fps: float,
    num_detections: int,
    model_name: str,
    latency_ms: float,
    tracking_enabled: bool = True,
    active_tracks: int = 0,
    ocr_texts_count: int = -1,
    sahi_enabled: bool = False,
    rag_enabled: bool = False,
    memory_enabled: bool = False,
    ann_version: Optional[str] = None,
    voice_status: str = "OFF",
    gesture_mode: str = "ALL_OBJECTS",
    active_gesture: str = "none",
    last_subtitle: Optional[str] = None,
) -> np.ndarray:
    """Renders a sleek HUD status bar and bottom subtitle display."""
    h, w = image.shape[:2]

    # 1. Semi-transparent top banner
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (20, 20, 20), cv2.FILLED)

    track_str = f"Tracks: {active_tracks}" if tracking_enabled else "Tracking: OFF"
    ocr_str = f" | OCR: {ocr_texts_count}" if ocr_texts_count >= 0 else ""
    sahi_str = " | SAHI: ON" if sahi_enabled else ""
    rag_str = " | RAG: ON" if rag_enabled else ""
    mem_str = " | Mem: ON" if memory_enabled else ""
    ann_str = f" | ANN: {ann_version}" if ann_version else " | ANN: (Fallback)"
    voice_badge = f" | Voice: {voice_status}" if voice_status != "OFF" else ""
    gesture_badge = f" | 🖐️ {gesture_mode}" if gesture_mode != "ALL_OBJECTS" else ""

    hud_text = (
        f"AURA v0.8 | {fps:5.1f} FPS | Infer: {latency_ms:4.1f}ms | "
        f"Detections: {num_detections} | {track_str}{sahi_str}{rag_str}{mem_str}{ocr_str}{ann_str}{voice_badge}{gesture_badge}"
    )

    # 2. Bottom subtitle banner if there is active speech or response
    if last_subtitle:
        banner_h = 44
        cv2.rectangle(overlay, (0, h - banner_h), (w, h), (15, 15, 15), cv2.FILLED)

    cv2.addWeighted(overlay, 0.78, image, 0.22, 0, image)

    # Render top text
    cv2.putText(
        image,
        hud_text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 255, 180),
        1,
        cv2.LINE_AA,
    )

    # Render bottom subtitle
    if last_subtitle:
        display_sub = last_subtitle[:100] + ("..." if len(last_subtitle) > 100 else "")
        cv2.putText(
            image,
            f"AURA: {display_sub}",
            (15, h - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return image


def run_pipeline(
    source_val: str,
    model_name: str,
    conf_thresh: float,
    device: str,
    width: int,
    height: int,
    headless: bool = False,
    benchmark: bool = False,
    max_frames: Optional[int] = None,
    dataset_csv: Optional[str] = None,
    no_track: bool = False,
    enable_ocr: bool = False,
    ocr_stride: int = 30,
    no_ann: bool = False,
    ann_model: str = "models/reliability_ann.pth",
    ann_scaler: str = "models/scaler.pkl",
    ann_thresh: float = 0.5,
    custom_classes: Optional[str] = None,
    sync_mode: bool = False,
    enable_voice: bool = False,
    enable_tts: bool = True,
    one_shot_query: Optional[str] = None,
    enable_chat_loop: bool = False,
    enable_sahi: bool = False,
    slice_size: int = 320,
    slice_overlap: float = 0.20,
    enable_rag: bool = False,
    rag_dir: str = "data/manuals",
    enable_memory: bool = False,
    memory_db: str = "data/memory.db",
    llm_provider: str = "offline",
    enable_gestures: bool = True,
) -> int:
    """Main execution loop for AURA Milestone 8 (Vision + SAHI + Memory + RAG + Voice + Gestures)."""
    source_target = int(source_val) if source_val.isdigit() else source_val
    classes_list = [c.strip() for c in custom_classes.split(",") if c.strip()] if custom_classes else None

    # SAHI Configuration
    sahi_config = SAHIConfig(
        enabled=enable_sahi,
        slice_width=slice_size,
        slice_height=slice_size,
        overlap_width_ratio=slice_overlap,
        overlap_height_ratio=slice_overlap,
    )

    # 1. Initialize ObjectDetector
    logger.info("Initializing ObjectDetector...")
    try:
        detector = ObjectDetector(
            model_name=model_name,
            confidence_threshold=conf_thresh,
            device=device,
            custom_classes=classes_list,
            sahi_config=sahi_config,
        )
    except ModelLoadError as e:
        logger.error(f"Detector initialization failed: {e}")
        return 1

    # 2. Initialize Tracker
    tracker: Optional[ObjectTracker] = None
    if not no_track:
        logger.info("Initializing Multi-Object Tracker...")
        tracker = ObjectTracker(max_age=30, min_hits=1, iou_threshold=0.3, smooth_factor=0.65)

    # 3. Initialize OCR Engine
    ocr_engine: Optional[OCREngine] = None
    if enable_ocr:
        logger.info("Initializing OCR Engine...")
        try:
            ocr_engine = OCREngine(confidence_threshold=0.3)
        except Exception as e:
            logger.warning(f"OCR Engine failed to initialize: {e}. Running without OCR.")

    # 4. Initialize Feature Builder & Reliability ANN
    feature_builder = FeatureBuilder()
    logger.info("Initializing Reliability ANN...")
    reliability_ann = ReliabilityInference(
        enabled=not no_ann,
        model_path=ann_model,
        scaler_path=ann_scaler,
        confidence_threshold=ann_thresh,
        device=device,
    )

    # 5. Initialize Context Manager, Episodic Memory, RAG Engine, and LLM Provider (Milestone 8)
    logger.info("Initializing Context Manager, Memory, RAG & LLM Engine...")
    context_manager = ContextManager()
    knowledge_retriever = KnowledgeRetriever(enable_curated=True, enable_wikipedia=True)

    rag_engine = RAGEngine(RAGConfig(enabled=enable_rag, docs_directory=rag_dir))
    if enable_rag:
        rag_engine.initialize()

    episodic_memory = EpisodicMemory(MemoryConfig(enabled=enable_memory, db_path=memory_db))

    intel_cfg = IntelligenceConfig(llm_provider=llm_provider)
    llm_reasoner = create_llm_provider(intel_cfg)

    conversation_engine = ConversationEngine(
        context_manager=context_manager,
        knowledge_retriever=knowledge_retriever,
        rag_engine=rag_engine,
        memory=episodic_memory,
        llm_provider=llm_reasoner,
    )

    # Subtitle state for HUD
    current_subtitle: Optional[str] = None
    subtitle_lock = threading.Lock()

    def update_subtitle(text: str):
        nonlocal current_subtitle
        with subtitle_lock:
            current_subtitle = text

    # 6. Initialize Voice Assistant (Milestone 7)
    voice_assistant: Optional[VoiceAssistant] = None
    if enable_voice:
        logger.info("Initializing Voice Assistant (STT & TTS)...")
        voice_assistant = VoiceAssistant(
            conversation_engine=conversation_engine,
            enable_tts=enable_tts,
            on_speech_start=lambda txt: update_subtitle(txt),
            on_speech_end=lambda txt: None,
        )

    # Gesture Action Callback Hooks
    def on_inspect_target(target: Detection):
        query = f"Inspect and describe this {target.class_name} in the scene."
        if voice_assistant:
            resp = voice_assistant.process_text_query(query, speak_output=enable_tts)
        else:
            resp = conversation_engine.respond(query)
        update_subtitle(f"AURA: {resp.response_text}")

    def on_toggle_sahi():
        if detector.sahi_config:
            detector.sahi_config.enabled = not detector.sahi_config.enabled
            logger.info(f"SAHI toggled to {detector.sahi_config.enabled}")

    def on_voice_trigger():
        if voice_assistant:
            threading.Thread(target=voice_assistant.listen_and_respond, daemon=True).start()

    gesture_controller = GestureActionController(
        debounce_frames=2,
        on_inspect_callback=on_inspect_target,
        on_toggle_sahi_callback=on_toggle_sahi,
        on_voice_trigger_callback=on_voice_trigger,
    ) if enable_gestures else None

    collector = DatasetCollector() if dataset_csv else None

    # 7. Initialize CameraAdapter
    use_synthetic = False
    camera: Optional[CameraAdapter] = None

    if source_val.lower() == "synthetic":
        use_synthetic = True
        logger.info("Using synthetic frame generator mode.")
    else:
        try:
            camera = CameraAdapter(
                source=source_target,
                width=width,
                height=height,
            )
            camera.open()
            logger.info(f"Connected to camera source '{source_target}'.")
        except CameraNotFoundError as e:
            if benchmark or headless:
                logger.warning(f"{e} - Falling back to synthetic frame generator.")
                use_synthetic = True
            else:
                logger.error(f"Cannot access camera: {e}")
                logger.info("Tip: Pass '--source synthetic' or '--headless --benchmark' to run without camera.")
                return 1

    window_name = "AURA - Multimodal Visual Agent"
    if not headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    total_latency = 0.0
    start_time = time.time()
    fps_ema = 0.0
    alpha = 0.1

    last_ocr_texts: List[TextDetection] = []
    tracking_active = not no_track

    # --- Decoupled Real-Time Inference Threading Setup ---
    use_async = (not sync_mode) and (not use_synthetic) and (not benchmark)
    
    current_detections: List[Detection] = []
    det_lock = threading.Lock()
    latest_camera_frame: Optional[Frame] = None
    frame_lock = threading.Lock()
    running = True
    last_infer_latency = 0.0

    def inference_worker():
        nonlocal current_detections, last_infer_latency
        while running:
            work_frame = None
            with frame_lock:
                if latest_camera_frame is not None:
                    work_frame = latest_camera_frame
            if work_frame is not None:
                t0_inf = time.perf_counter()
                dets = detector.detect(work_frame)
                t1_inf = time.perf_counter()
                with det_lock:
                    current_detections = dets
                    last_infer_latency = (t1_inf - t0_inf) * 1000.0
            time.sleep(0.005)

    ocr_lock = threading.Lock()
    def ocr_worker():
        nonlocal last_ocr_texts
        while running:
            if ocr_engine is not None:
                work_frame = None
                with frame_lock:
                    if latest_camera_frame is not None:
                        work_frame = latest_camera_frame
                if work_frame is not None:
                    try:
                        texts = ocr_engine.extract_text(work_frame)
                        with ocr_lock:
                            last_ocr_texts = texts
                    except Exception as e:
                        logger.debug(f"Async OCR error: {e}")
            time.sleep(1.5)

    if use_async:
        logger.info("Starting Decoupled Real-Time Async Vision Worker...")
        infer_thread = threading.Thread(target=inference_worker, daemon=True)
        infer_thread.start()
        if ocr_engine is not None:
            ocr_thread = threading.Thread(target=ocr_worker, daemon=True)
            ocr_thread.start()

    logger.info("Starting visual intelligence loop. Press 'q' or ESC in window to exit...")
    logger.info("Controls: 'v'=Voice, 'c'=Console Query, 'r'=RAG Lookup, 'k'=Knowledge, 'm'=Memory, 'i'=Scene Info, 't'=Tracker, 'h'=SAHI, 'o'=OCR")

    try:
        while True:
            t0 = time.perf_counter()

            # 1. Capture Frame
            if use_synthetic:
                frame = create_synthetic_frame(frame_count)
            else:
                frame = camera.read()
                if frame is None:
                    logger.info("End of video stream or frame capture returned None.")
                    break

            t_cap = time.perf_counter()

            # 2. Inference: Async or Sync
            if use_async:
                with frame_lock:
                    latest_camera_frame = frame
                with det_lock:
                    detections = list(current_detections)
                infer_latency_ms = last_infer_latency
            else:
                detections = detector.detect(frame)
                t_infer = time.perf_counter()
                infer_latency_ms = (t_infer - t_cap) * 1000.0

            total_latency += infer_latency_ms

            # 3. Multi-Object Tracking & Smoothing
            tracks_map: Dict[int, Any] = {}
            if tracking_active and tracker is not None:
                detections = tracker.update(detections)
                tracks_map = {t.track_id: t for t in tracker.all_tracks}

            # 4. Feature Extraction & Reliability Estimation
            features = feature_builder.extract_all(frame, detections, tracks_map=tracks_map) if detections else []
            if features:
                for i, feat in enumerate(features):
                    rel_res = reliability_ann.predict(feat)
                    detections[i].reliability_score = rel_res.score
                    detections[i].reliability_label = rel_res.label

            # 5. Non-blocking OCR state sync
            with ocr_lock:
                current_ocr_texts = list(last_ocr_texts)

            # 6. Update Multimodal Context & Object-Text associations
            object_texts_map: Dict[int, List[TextDetection]] = {}
            if ocr_engine is not None and current_ocr_texts:
                object_texts_map = ocr_engine.extract_text_for_detections(current_ocr_texts, detections)

            scene_context: SceneContext = context_manager.update(
                detections=detections,
                text_detections=current_ocr_texts,
                object_texts=object_texts_map,
                frame_shape=frame.shape,
            )

            # 7. Record Episodic Spatial Memory
            if enable_memory and episodic_memory:
                episodic_memory.record_scene(scene_context)

            # One-shot query handling if provided on CLI
            if one_shot_query and frame_count == 1:
                logger.info(f"Executing one-shot query: '{one_shot_query}'")
                if voice_assistant:
                    resp = voice_assistant.process_text_query(one_shot_query, speak_output=enable_tts)
                else:
                    resp = conversation_engine.respond(one_shot_query)
                print(f"\n[AURA Response to: '{one_shot_query}']\n=> {resp.response_text}\n")
                update_subtitle(resp.response_text)

            # 8. Record to Dataset if requested
            if collector is not None and features:
                collector.add_samples(features)

            # 9. Calculate Smooth FPS
            loop_time = time.perf_counter() - t0
            current_fps = 1.0 / max(loop_time, 1e-6)
            fps_ema = current_fps if frame_count == 0 else (alpha * current_fps + (1 - alpha) * fps_ema)

            # 10. Hand Gesture Interactive Control
            gesture_mode_val = "ALL_OBJECTS"
            active_gesture_val = "none"
            visible_detections = detections

            if gesture_controller is not None:
                (
                    visible_detections,
                    g_mode,
                    act_gest,
                    pointed_target,
                ) = gesture_controller.update(frame.image, detections)
                gesture_mode_val = g_mode.value
                active_gesture_val = act_gest.gesture.value

            # 11. Render Annotations & Enhanced HUD
            if gesture_mode_val == "HIDE_BOXES":
                annotated_frame = frame.image.copy()
            else:
                annotated_frame = detector.annotate(frame, visible_detections)
                if ocr_engine is not None and current_ocr_texts:
                    annotated_frame = draw_text_annotations(annotated_frame, current_ocr_texts)

                # Pointing Laser & Lock-On Visual Target Indicator
                if (
                    gesture_mode_val in ("FOCUS_OBJECT", "INSPECT")
                    and pointed_target is not None
                    and act_gest.pointing_tip is not None
                ):
                    ptx, pty = int(act_gest.pointing_tip[0]), int(act_gest.pointing_tip[1])
                    tx1, ty1, tx2, ty2 = [int(c) for c in pointed_target.bbox]
                    tcx, tcy = (tx1 + tx2) // 2, (ty1 + ty2) // 2
                    
                    # Draw laser ray with glow
                    cv2.line(annotated_frame, (ptx, pty), (tcx, tcy), (0, 240, 255), 2, cv2.LINE_AA)
                    cv2.circle(annotated_frame, (ptx, pty), 7, (0, 240, 255), -1, cv2.LINE_AA)
                    
                    # Target Crosshair Reticle
                    cv2.circle(annotated_frame, (tcx, tcy), 14, (0, 255, 120), 2, cv2.LINE_AA)
                    cv2.drawMarker(annotated_frame, (tcx, tcy), (0, 255, 120), cv2.MARKER_CROSS, 20, 2)
                    
                    tag_prefix = "👌 INSPECTING" if gesture_mode_val == "INSPECT" else "👉 LOCKED"
                    cv2.putText(
                        annotated_frame,
                        f"{tag_prefix}: {pointed_target.class_name.upper()}",
                        (tx1, max(24, ty1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.60,
                        (0, 240, 255),
                        2,
                    )

            # Draw Cybernetic 21-Landmark Hand Skeleton
            if act_gest.landmarks:
                annotated_frame = draw_hand_skeleton(annotated_frame, act_gest)

            # Draw Real-Time HUD Action Toast Banner
            if gesture_controller and gesture_controller.active_toast and time.time() < gesture_controller.toast_expiry_time:
                annotated_frame = draw_action_toast(annotated_frame, gesture_controller.active_toast)

            active_tracks_count = len(tracker.active_tracks) if (tracking_active and tracker) else 0
            ocr_count = len(current_ocr_texts) if ocr_engine is not None else -1
            v_status = voice_assistant.status if voice_assistant else ("IDLE" if enable_voice else "OFF")
            sahi_is_active = bool(detector.sahi_config and detector.sahi_config.enabled)

            with subtitle_lock:
                sub_text = current_subtitle

            annotated_frame = draw_hud(
                annotated_frame,
                fps=fps_ema,
                num_detections=len(visible_detections),
                model_name=detector.model_name,
                latency_ms=infer_latency_ms,
                tracking_enabled=tracking_active,
                active_tracks=active_tracks_count,
                ocr_texts_count=ocr_count,
                sahi_enabled=sahi_is_active,
                rag_enabled=enable_rag,
                memory_enabled=enable_memory,
                ann_version=reliability_ann.model_version,
                voice_status=v_status,
                gesture_mode=gesture_mode_val,
                active_gesture=active_gesture_val,
                last_subtitle=sub_text,
            )

            frame_count += 1

            if frame_count % 30 == 0 or frame_count == 1:
                logger.info(
                    f"Frame {frame_count:4d} | FPS: {fps_ema:5.1f} | Latency: {infer_latency_ms:5.1f}ms | Detections: {len(detections)} | Tracks: {active_tracks_count} | Texts: {max(0, ocr_count)}"
                )

            # 11. Display GUI if not headless
            if not headless:
                cv2.imshow(window_name, annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    logger.info("Exit key pressed by user.")
                    break
                elif key in (ord('v'), ord('V')):
                    if voice_assistant:
                        logger.info("Triggering Voice Push-To-Talk...")
                        voice_assistant.trigger_push_to_talk()
                    else:
                        logger.info("Voice assistant is not enabled. Pass '--voice' to enable.")
                elif key in (ord('c'), ord('C')):
                    # Console query
                    print("\n[AURA Prompt] Enter query about visual scene / manuals / memory: ", end="", flush=True)
                    user_q = input().strip()
                    if user_q:
                        if voice_assistant:
                            resp = voice_assistant.process_text_query(user_q, speak_output=enable_tts)
                        else:
                            resp = conversation_engine.respond(user_q)
                        print(f"[AURA Answer] {resp.response_text}\n")
                        update_subtitle(resp.response_text)
                elif key in (ord('r'), ord('R')):
                    # RAG Document Lookup on focused object or query
                    if detections:
                        target = detections[0]
                        rag_res = rag_engine.retrieve_for_detection(target)
                        if rag_res.has_results:
                            top_doc = rag_res.top_document
                            print(f"\n[RAG Document Match: {top_doc.title}]\n{top_doc.content}\n")
                            update_subtitle(f"Manual: {top_doc.title}")
                        else:
                            print(f"\n[RAG] No matching manuals found for '{target.class_name}'.\n")
                    else:
                        print("\n[RAG] No objects detected to match against manuals.\n")
                elif key in (ord('k'), ord('K')):
                    # Object Knowledge Lookup
                    if detections:
                        target = detections[0]
                        k_item = knowledge_retriever.retrieve_for_detection(target)
                        if k_item:
                            print(f"\n[Knowledge: {k_item.title}]\nCategory: {k_item.category}\nSummary: {k_item.summary}\nSource: {k_item.source}\n")
                            update_subtitle(f"{k_item.title}: {k_item.summary}")
                            if voice_assistant and enable_tts:
                                voice_assistant.tts.speak(k_item.summary)
                        else:
                            print(f"\n[Knowledge] No encyclopedia facts found for '{target.class_name}'.\n")
                    else:
                        print("\n[Knowledge] No objects detected in current view.\n")
                elif key in (ord('m'), ord('M')):
                    if enable_memory and episodic_memory:
                        print("\n[Episodic Memory - Recent Events]")
                        events = episodic_memory.get_history("face", limit=3) + episodic_memory.get_history("person", limit=3) + episodic_memory.get_history("laptop", limit=3)
                        for ev in events:
                            print(f" - {ev.describe()}")
                        print()
                    else:
                        print("\n[Episodic Memory] Disabled. Pass '--memory' to enable.\n")
                elif key in (ord('i'), ord('I')):
                    print(f"\n[Scene Context Summary]\n{scene_context.summary()}\n")
                    for e in scene_context.entities:
                        print(f" - {e.describe()}")
                    print()
                elif key in (ord('t'), ord('T')):
                    tracking_active = not tracking_active
                    logger.info(f"Tracking toggled: {'ENABLED' if tracking_active else 'DISABLED'}")
                elif key in (ord('h'), ord('H')):
                    if detector.sahi_config and detector.sahi_config.enabled:
                        detector.disable_sahi()
                        logger.info("SAHI sliced inference: DISABLED")
                    else:
                        detector.enable_sahi(SAHIConfig(enabled=True, slice_width=slice_size, slice_height=slice_size, overlap_width_ratio=slice_overlap, overlap_height_ratio=slice_overlap))
                        logger.info("SAHI sliced inference: ENABLED")
                elif key in (ord('o'), ord('O')):
                    logger.info(f"Triggering instant OCR scan on frame {frame_count}...")
                    if ocr_engine is None:
                        ocr_engine = OCREngine(confidence_threshold=0.3)
                    last_ocr_texts = ocr_engine.extract_text(frame)
                    logger.info(f"Extracted {len(last_ocr_texts)} text instances:")
                    for td in last_ocr_texts:
                        print(f"   - '{td.text}' ({int(td.confidence * 100)}% conf) at {td.bbox}")
                elif key in (ord('s'), ord('S')):
                    filename = f"aura_capture_{int(time.time())}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    logger.info(f"Saved snapshot to '{filename}'")

            # Check max frames constraint
            if max_frames is not None and frame_count >= max_frames:
                logger.info(f"Reached specified frame limit ({max_frames} frames). Exiting loop.")
                break

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Stopping...")
    finally:
        running = False
        if voice_assistant:
            voice_assistant.shutdown()
        if enable_memory and episodic_memory:
            episodic_memory.close()
        total_time = time.time() - start_time
        avg_fps = frame_count / max(total_time, 1e-6)
        avg_latency = total_latency / max(frame_count, 1)

        logger.info("=" * 60)
        logger.info("AURA Multimodal Agent Execution Summary:")
        logger.info(f"  Total frames processed : {frame_count}")
        logger.info(f"  Total elapsed time     : {total_time:.2f} s")
        logger.info(f"  Average throughput     : {avg_fps:.1f} FPS")
        logger.info(f"  Average infer latency  : {avg_latency:.2f} ms")
        if collector is not None and collector.count > 0:
            collector.to_csv(dataset_csv)
            logger.info(f"  Dataset samples saved  : {collector.count} to '{dataset_csv}'")
        if camera is not None:
            camera.release()
        if not headless:
            cv2.destroyAllWindows()

    return 0


def main():
    args = parse_args()
    return run_pipeline(
        source_val=args.source,
        model_name=args.model,
        conf_thresh=args.conf,
        device=args.device,
        width=args.width,
        height=args.height,
        headless=args.headless or args.benchmark,
        benchmark=args.benchmark,
        max_frames=args.frames if args.frames is not None else (50 if args.benchmark else None),
        dataset_csv=args.collect_dataset,
        no_track=args.no_track,
        enable_ocr=args.ocr,
        ocr_stride=args.ocr_stride,
        no_ann=args.no_ann,
        ann_model=args.ann_model,
        ann_scaler=args.ann_scaler,
        ann_thresh=args.ann_thresh,
        custom_classes=args.classes,
        sync_mode=args.sync,
        enable_voice=args.voice,
        enable_tts=not args.no_tts,
        one_shot_query=args.query,
        enable_chat_loop=args.chat,
        enable_sahi=args.sahi,
        slice_size=args.slice_size,
        slice_overlap=args.slice_overlap,
        enable_rag=args.rag,
        rag_dir=args.rag_dir,
        enable_memory=args.memory,
        memory_db=args.memory_db,
        llm_provider=args.llm,
        enable_gestures=args.gestures,
    )


if __name__ == "__main__":
    sys.exit(main())
