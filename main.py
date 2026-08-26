"""
AURA - Adaptive Understanding and Reasoning Architecture
Milestone 5: Multi-Object Tracking & Optical Character Recognition (OCR) Pipeline
(Camera -> OpenCV -> YOLO Detection -> Tracker -> OCR -> FeatureBuilder -> Reliability ANN -> VisionPipeline)

Entry point for running live detection, tracking, OCR, feature extraction, dataset collection, or benchmarking.
"""

import sys
import time
import argparse
import logging
from typing import Optional, List, Dict, Any
import cv2
import numpy as np

from vision.camera import CameraAdapter, CameraNotFoundError, Frame
from vision.detector import ObjectDetector, ModelLoadError
from vision.tracker import ObjectTracker
from vision.features import FeatureBuilder
from vision.dataset_collector import DatasetCollector
from ocr.engine import OCREngine, TextDetection, draw_text_annotations

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
        default="yolo11m.pt",
        help="YOLO model path or identifier (e.g., 'yolo11m.pt', 'yolo11s.pt', 'yolo11n.pt'). Default: 'yolo11m.pt'",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.40,
        help="Detection confidence threshold in range [0.0, 1.0]. Default: 0.40",
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
        help="Run without displaying OpenCV GUI window (useful for headless servers or automated testing).",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark mode to measure throughput (FPS) and inference latency.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Stop automatically after processing N frames.",
    )
    parser.add_argument(
        "--collect-dataset",
        type=str,
        default=None,
        help="Path to CSV file where extracted detection features should be recorded (e.g., 'data/features.csv').",
    )
    parser.add_argument(
        "--no-track",
        action="store_true",
        help="Disable multi-object tracking (persistent track IDs).",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Enable Optical Character Recognition (OCR) extraction.",
    )
    parser.add_argument(
        "--ocr-stride",
        type=int,
        default=15,
        help="Perform full OCR scan every N frames to maintain high video FPS. Default: 15",
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
    return parser.parse_args()


def create_synthetic_frame(frame_idx: int = 0) -> Frame:
    """Generates a synthetic test frame with moving shapes and visible text for benchmarking."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw background gradient
    for y in range(480):
        img[y, :, :] = (int(30 + 30 * (y / 480)), int(25 + 20 * (y / 480)), int(40 + 30 * (y / 480)))
    
    # Draw moving geometric shapes
    offset_x = int((frame_idx * 5) % 400)
    cv2.rectangle(img, (50 + offset_x, 100), (180 + offset_x, 260), (0, 200, 255), -1)
    cv2.circle(img, (400, 240), 60, (255, 100, 0), -1)
    
    # Clear readable text
    cv2.putText(
        img,
        f"AURA OCR SYSTEM",
        (160, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"Frame: {frame_idx:04d}",
        (20, 440),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 200),
        2,
        cv2.LINE_AA,
    )
    return Frame(image=img, timestamp=time.time(), source_id="synthetic_generator")


def draw_hud(
    image: np.ndarray,
    fps: float,
    num_detections: int,
    model_name: str,
    latency_ms: float,
    tracking_enabled: bool,
    active_tracks: int,
    ocr_texts_count: int,
    ann_version: str = "disabled",
) -> np.ndarray:
    """Draws a sleek Head-Up-Display (HUD) overlay on the top edge of the frame."""
    hud_h = 38
    hud_bg = image[:hud_h, :].astype(np.float32)
    # Dark translucent header bar
    dark_overlay = np.zeros_like(hud_bg)
    dark_overlay[:] = (20, 20, 25)
    image[:hud_h, :] = cv2.addWeighted(hud_bg, 0.35, dark_overlay, 0.65, 0).astype(np.uint8)

    # Info string
    track_str = f"Tracks: {active_tracks}" if tracking_enabled else "Track: OFF"
    ocr_str = f"OCR: {ocr_texts_count}" if ocr_texts_count >= 0 else "OCR: OFF"
    hud_text = f"AURA M5 | FPS: {fps:4.1f} | {latency_ms:4.1f}ms | Obj: {num_detections} | {track_str} | {ocr_str} | ANN: {ann_version}"
    cv2.putText(
        image,
        hud_text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
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
    ocr_stride: int = 15,
    no_ann: bool = False,
    ann_model: str = "models/reliability_ann.pth",
    ann_scaler: str = "models/scaler.pkl",
    ann_thresh: float = 0.5,
) -> int:
    """Main execution loop for AURA Milestone 5."""
    source_target = int(source_val) if source_val.isdigit() else source_val

    # Initialize ObjectDetector
    logger.info("Initializing ObjectDetector...")
    try:
        detector = ObjectDetector(
            model_name=model_name,
            confidence_threshold=conf_thresh,
            device=device,
        )
    except ModelLoadError as e:
        logger.error(f"Detector initialization failed: {e}")
        return 1

    # Initialize Tracker
    tracker: Optional[ObjectTracker] = None
    if not no_track:
        logger.info("Initializing Multi-Object Tracker...")
        tracker = ObjectTracker(max_age=30, min_hits=1, iou_threshold=0.3)

    # Initialize OCR Engine
    ocr_engine: Optional[OCREngine] = None
    if enable_ocr:
        logger.info("Initializing OCR Engine...")
        ocr_engine = OCREngine(confidence_threshold=0.3)

    feature_builder = FeatureBuilder()
    
    # Initialize Reliability ANN
    from ann.inference import ReliabilityInference
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
                logger.warning(f"{e} - Falling back to synthetic frame generator for testing/benchmarking.")
                use_synthetic = True
            else:
                logger.error(f"Cannot access camera: {e}")
                logger.info("Tip: Pass '--source synthetic' or '--headless --benchmark' to run automated testing without a camera.")
                return 1

    window_name = "AURA - Milestone 5 (Tracking & OCR)"
    if not headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    total_latency = 0.0
    start_time = time.time()
    fps_ema = 0.0
    alpha = 0.1  # Smoothing factor for FPS

    last_ocr_texts: List[TextDetection] = []
    tracking_active = not no_track

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

            # 2. Run Object Detection
            detections = detector.detect(frame)

            # 3. Run Multi-Object Tracking
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

            t_infer = time.perf_counter()
            infer_latency_ms = (t_infer - t_cap) * 1000.0
            total_latency += infer_latency_ms

            # 6. Record to Dataset if requested
            if collector is not None and features:
                collector.add_samples(features)

            # 7. Calculate FPS
            loop_time = t_infer - t0
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
                model_name=model_name,
                latency_ms=infer_latency_ms,
                tracking_enabled=tracking_active,
                active_tracks=active_tracks_count,
                ocr_texts_count=ocr_count,
                ann_version=reliability_ann.model_version,
            )

            frame_count += 1

            # Log periodic stats
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
        logger.info("=" * 60)

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
    )


if __name__ == "__main__":
    sys.exit(main())
