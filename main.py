"""
AURA - Adaptive Understanding and Reasoning Architecture
Milestone 5: Multi-Object Tracking & Optical Character Recognition (OCR) Pipeline
(Camera -> OpenCV -> Decoupled Real-Time Threading -> YOLO Detection -> Tracker -> OCR -> Reliability ANN)

Entry point for running live detection, tracking, OCR, feature extraction, dataset collection, or benchmarking.
"""

import sys
import time
import argparse
import logging
import threading
from typing import Optional, List, Dict, Any
import cv2
import numpy as np

from vision.camera import CameraAdapter, CameraNotFoundError, Frame
from vision.detector import ObjectDetector, ModelLoadError, Detection
from vision.tracker import ObjectTracker
from vision.features import FeatureBuilder
from vision.dataset_collector import DatasetCollector
from ocr.engine import OCREngine, TextDetection, draw_text_annotations
from ann.inference import ReliabilityInference

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AURA.Main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AURA Visual Intelligence Assistant - Milestone 5 (Tracking & OCR)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Camera device index (e.g., '0') or path to video/image file. Default: '0'",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8m-worldv2.pt",
        help="YOLO model path (e.g., 'yolov8m-worldv2.pt', 'yolov8s-worldv2.pt', 'yolo11s.pt'). Default: 'yolov8m-worldv2.pt'",
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
        help="Comma-separated custom class names for YOLO-World (e.g. 'person,laptop,notebook,pen,smartphone'). Default: AURA indoor vocabulary",
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
        help="Run without displaying GUI window (useful for servers, background, testing).",
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
        default=0.5,
        help="Decision threshold for reliable/unreliable label. Default: 0.5",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Force synchronous single-threaded execution (disables decoupled real-time preview thread).",
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
    cv2.putText(img, "AURA NOTEBOOK", (nx - 55, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

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
    ann_version: Optional[str] = None,
) -> np.ndarray:
    """Renders a sleek HUD status bar over the frame."""
    h, w = image.shape[:2]
    # Semi-transparent top banner
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)

    track_str = f"Tracks: {active_tracks}" if tracking_enabled else "Tracking: OFF"
    ocr_str = f" | OCR: {ocr_texts_count}" if ocr_texts_count >= 0 else ""
    ann_str = f" | ANN: {ann_version}" if ann_version else " | ANN: (Fallback)"

    hud_text = (
        f"AURA v0.5 | {fps:5.1f} FPS | Infer: {latency_ms:4.1f}ms | "
        f"Detections: {num_detections} | {track_str}{ocr_str}{ann_str}"
    )

    cv2.putText(
        image,
        hud_text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 255, 180),
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
) -> int:
    """Main execution loop for AURA Milestone 5."""
    source_target = int(source_val) if source_val.isdigit() else source_val
    classes_list = [c.strip() for c in custom_classes.split(",") if c.strip()] if custom_classes else None

    # Initialize ObjectDetector
    logger.info("Initializing ObjectDetector...")
    try:
        detector = ObjectDetector(
            model_name=model_name,
            confidence_threshold=conf_thresh,
            device=device,
            custom_classes=classes_list,
        )
    except ModelLoadError as e:
        logger.error(f"Detector initialization failed: {e}")
        return 1

    # Initialize Tracker
    tracker: Optional[ObjectTracker] = None
    if not no_track:
        logger.info("Initializing Multi-Object Tracker...")
        tracker = ObjectTracker(max_age=30, min_hits=1, iou_threshold=0.3, smooth_factor=0.65)

    # Initialize OCR Engine
    ocr_engine: Optional[OCREngine] = None
    if enable_ocr:
        logger.info("Initializing OCR Engine...")
        try:
            ocr_engine = OCREngine(confidence_threshold=0.3)
        except Exception as e:
            logger.warning(f"OCR Engine failed to initialize: {e}. Running without OCR.")

    # Initialize Feature Builder
    feature_builder = FeatureBuilder()

    # Initialize Reliability ANN
    logger.info("Initializing Reliability ANN...")
    reliability_ann = ReliabilityInference(
        enabled=not no_ann,
        model_path=ann_model,
        scaler_path=ann_scaler,
        confidence_threshold=ann_thresh,
        device=device,
    )

    collector = DatasetCollector() if dataset_csv else None

    # Initialize CameraAdapter
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

    window_name = "AURA - High-Precision Assistant"
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

    infer_thread = None
    if use_async:
        logger.info("Starting Decoupled Real-Time Async Vision Worker (30+ FPS Preview)...")
        infer_thread = threading.Thread(target=inference_worker, daemon=True)
        infer_thread.start()

    logger.info("Starting visual processing loop. Press 'q' or ESC in window to exit...")
    logger.info("Controls: 'q'=Quit, 's'=Save Snapshot, 'c'=Print Detections, 'f'=Print Features, 't'=Toggle Tracker, 'o'=Scan OCR")

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

            # 5. Periodic / Strided OCR
            if ocr_engine is not None and (frame_count % max(1, ocr_stride) == 0):
                last_ocr_texts = ocr_engine.extract_text(frame)

            # 6. Record to Dataset if requested
            if collector is not None and features:
                collector.add_samples(features)

            # 7. Calculate Smooth FPS
            loop_time = time.perf_counter() - t0
            current_fps = 1.0 / max(loop_time, 1e-6)
            fps_ema = current_fps if frame_count == 0 else (alpha * current_fps + (1 - alpha) * fps_ema)

            # 8. Render Annotations & HUD
            annotated_frame = detector.annotate(frame, detections)
            if ocr_engine is not None and last_ocr_texts:
                annotated_frame = draw_text_annotations(annotated_frame, last_ocr_texts)

            active_tracks_count = len(tracker.active_tracks) if (tracking_active and tracker) else 0
            ocr_count = len(last_ocr_texts) if ocr_engine is not None else -1

            annotated_frame = draw_hud(
                annotated_frame,
                fps=fps_ema,
                num_detections=len(detections),
                model_name=detector.model_name,
                latency_ms=infer_latency_ms,
                tracking_enabled=tracking_active,
                active_tracks=active_tracks_count,
                ocr_texts_count=ocr_count,
                ann_version=reliability_ann.model_version,
            )

            frame_count += 1

            if frame_count % 30 == 0 or frame_count == 1:
                logger.info(
                    f"Frame {frame_count:4d} | FPS: {fps_ema:5.1f} | Latency: {infer_latency_ms:5.1f}ms | Detections: {len(detections)} | Tracks: {active_tracks_count} | Texts: {max(0, ocr_count)}"
                )

            # 9. Display GUI if not headless
            if not headless:
                cv2.imshow(window_name, annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    logger.info("Exit key pressed by user.")
                    break
                elif key in (ord('c'), ord('C')):
                    logger.info(f"Structured detections at frame {frame_count}:")
                    for d in detections:
                        print(" ", d.to_dict())
                elif key in (ord('f'), ord('F')):
                    logger.info(f"Extracted features at frame {frame_count}:")
                    for feat in features:
                        print(" ", feat.to_dict())
                elif key in (ord('t'), ord('T')):
                    tracking_active = not tracking_active
                    logger.info(f"Tracking toggled: {'ENABLED' if tracking_active else 'DISABLED'}")
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
        total_time = time.time() - start_time
        avg_fps = frame_count / max(total_time, 1e-6)
        avg_latency = total_latency / max(frame_count, 1)

        logger.info("=" * 60)
        logger.info("AURA Milestone 5 Execution Summary:")
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
    )


if __name__ == "__main__":
    sys.exit(main())
