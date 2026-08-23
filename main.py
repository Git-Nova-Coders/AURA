"""
AURA - Adaptive Understanding and Reasoning Architecture
Milestone 1: Real-Time Computer Vision Pipeline (Camera -> OpenCV -> YOLO Detection)

Entry point for running live detection, benchmarking, or testing without GUI.
"""

import sys
import time
import argparse
import logging
from typing import Optional
import cv2
import numpy as np

from vision.camera import CameraAdapter, CameraNotFoundError, Frame
from vision.detector import ObjectDetector, ModelLoadError

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AURA.Main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AURA Visual Intelligence Assistant - Milestone 1"
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
        default="yolo11n.pt",
        help="YOLO model path or identifier (e.g., 'yolo11n.pt', 'yolov8n.pt'). Default: 'yolo11n.pt'",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold in range [0.0, 1.0]. Default: 0.25",
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
    return parser.parse_args()


def create_synthetic_frame(frame_idx: int = 0) -> Frame:
    """Generates a synthetic test frame for benchmark or headless verification without physical cameras."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw background gradient
    for y in range(480):
        img[y, :, :] = (int(30 + 30 * (y / 480)), int(25 + 20 * (y / 480)), int(40 + 30 * (y / 480)))
    
    # Draw moving geometric shapes
    offset_x = int((frame_idx * 5) % 400)
    cv2.rectangle(img, (50 + offset_x, 100), (180 + offset_x, 260), (0, 200, 255), -1)
    cv2.circle(img, (400, 240), 60, (255, 100, 0), -1)
    
    # Text label
    cv2.putText(
        img,
        f"AURA Synthetic Test Frame #{frame_idx}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
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
) -> np.ndarray:
    """Draws a sleek Head-Up-Display (HUD) overlay on the top edge of the frame."""
    hud_h = 36
    hud_bg = image[:hud_h, :].astype(np.float32)
    # Dark translucent header bar
    dark_overlay = np.zeros_like(hud_bg)
    dark_overlay[:] = (20, 20, 25)
    image[:hud_h, :] = cv2.addWeighted(hud_bg, 0.35, dark_overlay, 0.65, 0).astype(np.uint8)

    # Info string
    hud_text = f"AURA M1  |  FPS: {fps:5.1f}  |  Latency: {latency_ms:4.1f}ms  |  Detections: {num_detections:2d}  |  Model: {model_name}"
    cv2.putText(
        image,
        hud_text,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
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
) -> int:
    """Main execution loop for AURA Milestone 1."""
    # Parse source
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

    window_name = "AURA - Milestone 1 (OpenCV + YOLO)"
    if not headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    total_latency = 0.0
    start_time = time.time()
    fps_ema = 0.0
    alpha = 0.1  # Smoothing factor for FPS

    logger.info("Starting visual processing loop. Press 'q' or ESC in window to exit...")

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
            t_infer = time.perf_counter()
            infer_latency_ms = (t_infer - t_cap) * 1000.0
            total_latency += infer_latency_ms

            # 3. Calculate FPS
            loop_time = t_infer - t0
            current_fps = 1.0 / max(loop_time, 1e-6)
            fps_ema = current_fps if frame_count == 0 else (alpha * current_fps + (1 - alpha) * fps_ema)

            # 4. Render Annotations & HUD
            annotated_frame = detector.annotate(frame, detections)
            annotated_frame = draw_hud(
                annotated_frame,
                fps=fps_ema,
                num_detections=len(detections),
                model_name=model_name,
                latency_ms=infer_latency_ms,
            )

            frame_count += 1

            # Log periodic stats in headless/benchmark mode
            if frame_count % 30 == 0 or frame_count == 1:
                logger.info(
                    f"Frame {frame_count:4d} | FPS: {fps_ema:5.1f} | Latency: {infer_latency_ms:5.1f}ms | Detections: {len(detections)}"
                )

            # 5. Display GUI if not headless
            if not headless:
                cv2.imshow(window_name, annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    logger.info("Exit key pressed by user.")
                    break
                elif key in (ord('c'), ord('C')):
                    # Print structured detections to console
                    logger.info(f"Structured detections at frame {frame_count}:")
                    for d in detections:
                        print(" ", d.to_dict())
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
        logger.info(f"AURA Milestone 1 Execution Summary:")
        logger.info(f"  Total frames processed : {frame_count}")
        logger.info(f"  Total elapsed time     : {total_time:.2f} s")
        logger.info(f"  Average throughput     : {avg_fps:.1f} FPS")
        logger.info(f"  Average infer latency  : {avg_latency:.2f} ms")
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
    )


if __name__ == "__main__":
    sys.exit(main())
