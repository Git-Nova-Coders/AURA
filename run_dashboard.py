"""
AURA Dashboard Launcher
Standalone entry point to start the AURA vision pipeline and web dashboard.

Usage:
    python run_dashboard.py                          # Synthetic mode (no camera needed)
    python run_dashboard.py --source 0               # Webcam
    python run_dashboard.py --source video.mp4       # Video file
    python run_dashboard.py --port 8420 --host 0.0.0.0
"""

import sys
import time
import argparse
import logging
import webbrowser
import threading

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AURA.Dashboard")


def parse_args():
    parser = argparse.ArgumentParser(
        description="AURA Interactive Web Dashboard — Vision + AI Interface"
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="Camera index, video path, or 'synthetic'. Default: 0 (webcam)",
    )
    parser.add_argument(
        "--model", type=str, default="yolov8m-worldv2.pt",
        help="YOLO model path. Default: yolov8m-worldv2.pt",
    )
    parser.add_argument(
        "--conf", type=float, default=0.35,
        help="Detection confidence threshold. Default: 0.35",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Inference device. Default: auto",
    )
    parser.add_argument(
        "--classes", type=str, default=None,
        help="Comma-separated custom class names for YOLO-World.",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Server bind address. Default: 0.0.0.0",
    )
    parser.add_argument(
        "--port", type=int, default=8420,
        help="Server port. Default: 8420",
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="Don't auto-open the browser.",
    )
    parser.add_argument(
        "--sahi", action="store_true",
        help="Enable SAHI sliced inference.",
    )
    parser.add_argument(
        "--no-rag", action="store_true",
        help="Disable RAG document search.",
    )
    parser.add_argument(
        "--no-memory", action="store_true",
        help="Disable episodic memory.",
    )
    parser.add_argument(
        "--no-ocr", action="store_true",
        help="Disable OCR text extraction.",
    )
    parser.add_argument(
        "--llm", type=str, default="offline",
        help="LLM provider: offline, gemini, ollama, openai. Default: offline",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Instantly Launch Holographic Splash Screen (Sub-50ms)
    splash = None
    try:
        from interface.splash import launch_splash
        splash = launch_splash()
    except Exception as e:
        logger.warning(f"Splash screen fallback: {e}")

    logger.info("=" * 60)
    logger.info("  AURA — Adaptive Understanding & Reasoning Architecture")
    logger.info("  Interactive Web Dashboard v1.0.0")
    logger.info("=" * 60)

    if splash:
        splash.update_step(0.25, "LOADING NEURAL VISION & MODEL WEIGHTS...")

    # Import after arg parse to avoid slow import on --help
    from interface.bridge import AuraBridge
    from interface.server import run_server

    # Initialize the pipeline bridge
    logger.info("Initializing AURA Pipeline Bridge...")
    if splash:
        splash.update_step(0.50, "CALIBRATING 21-LANDMARK 3D GESTURE SENSORS...")

    try:
        bridge = AuraBridge(
            source=args.source,
            model_name=args.model,
            conf_thresh=args.conf,
            device=args.device,
            custom_classes=args.classes,
            enable_sahi=args.sahi,
            enable_ocr=not args.no_ocr,
            enable_rag=not args.no_rag,
            enable_memory=not args.no_memory,
            llm_provider=args.llm,
        )
    except Exception as e:
        if splash:
            splash.close()
        logger.error(f"Failed to initialize pipeline: {e}")
        return 1

    # Start the vision pipeline in background
    bridge.start()
    logger.info("Vision pipeline started.")

    if splash:
        splash.update_step(0.85, "SYNCHRONIZING WEBSOCKET TELEMETRY CHANNELS...")

    def open_browser_and_dismiss_splash():
        url = f"http://localhost:{args.port}"
        logger.info(f"Opening dashboard at {url}")
        if splash:
            splash.update_step(1.00, "SYSTEM READY // LAUNCHING COCKPIT...")
        webbrowser.open(url)
        if splash:
            time.sleep(0.8)
            splash.close()

    # Auto-open browser
    if not args.no_browser:
        threading.Timer(1.0, open_browser_and_dismiss_splash).start()
    elif splash:
        threading.Timer(0.8, splash.close).start()

    # Start the web server (blocking)
    try:
        run_server(bridge, host=args.host, port=args.port)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        if splash:
            splash.close()
        bridge.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
