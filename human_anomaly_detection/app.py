import argparse
import os
import sys
import time
import cv2

from src.anomaly_detector import AnomalyDetector
from src.person_detector import PersonDetector
import src.config as config

def run_image(image_path: str, person_detector: PersonDetector, anomaly_detector: AnomalyDetector, save_output: bool = False):
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        return

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not decode image: {image_path}")
        return

    annotated = process_frame(frame, person_detector, anomaly_detector)
    
    if save_output:
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        out_name = f"app_output_{os.path.basename(image_path)}"
        out_path = os.path.join(config.RESULTS_DIR, out_name)
        cv2.imwrite(out_path, annotated)
        print(f"Annotated result saved to: {out_path}")

    cv2.imshow("AURA - Human Anomaly Detection", annotated)
    print("Image displayed. Press any key on the image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_frame(frame, person_detector: PersonDetector, anomaly_detector: AnomalyDetector):
    annotated = frame.copy()
    
    # 1. Detect person bounding boxes
    person_results = person_detector.detect_and_crop(frame)
    
    # If no persons detected, run anomaly detector on whole frame as fallback
    if len(person_results) == 0:
        pred = anomaly_detector.predict(frame)
        label = pred.get("label", "Unknown")
        conf = pred.get("confidence", 0.0) * 100
        color = (0, 255, 0) if label == "Normal" else (0, 0, 255)
        text = f"Full Frame: {label} ({conf:.1f}%)"
        cv2.putText(annotated, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return annotated

    # 2. For each detected person crop, predict anomaly
    for item in person_results:
        crop = item["crop"]
        x1, y1, x2, y2 = item["box"]
        
        pred = anomaly_detector.predict(crop)
        label = pred.get("label", "Unknown")
        conf = pred.get("confidence", 0.0) * 100
        
        # Color coding: Green for Normal, Red for Anomaly
        color = (0, 255, 0) if label == "Normal" else (0, 0, 255)
        
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Draw label badge
        label_text = f"{label}: {conf:.1f}%"
        (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        badge_y1 = max(0, y1 - 25)
        cv2.rectangle(annotated, (x1, badge_y1), (x1 + w + 10, badge_y1 + h + 8), color, cv2.FILLED)
        cv2.putText(annotated, label_text, (x1 + 5, badge_y1 + h + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return annotated

def run_video_or_camera(source, person_detector: PersonDetector, anomaly_detector: AnomalyDetector):
    # Parse source as camera index if numeric
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video/camera source: {source}")
        return

    print(f"Streaming from source: {source}")
    print("Press 'q' to exit.")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video stream finished or frame unreadable.")
            break

        annotated = process_frame(frame, person_detector, anomaly_detector)

        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # Draw FPS counter
        cv2.putText(annotated, f"FPS: {fps:.1f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("AURA - Human Anomaly Detection Application", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Session ended by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Human Anomaly Detection Application Layer")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--video", type=str, help="Path to input video file or camera index (e.g. 0)")
    parser.add_argument("--camera", action="store_true", help="Launch live camera (webcam index 0)")
    parser.add_argument("--save", action="store_true", help="Save annotated output when running in image mode")
    args = parser.parse_args()

    model_path = os.path.join(config.MODELS_DIR, "anomaly_model.pth")
    if not os.path.exists(model_path):
        print(f"Error: Trained anomaly model not found at {model_path}.")
        print("Please train the model first using: python -m src.train")
        sys.exit(1)

    print("Loading models (Person Detector + Anomaly Detector)...")
    person_det = PersonDetector(confidence_threshold=0.6)
    anomaly_det = AnomalyDetector(model_path)
    print("Models successfully loaded and ready.")

    if args.image:
        run_image(args.image, person_det, anomaly_det, save_output=args.save)
    elif args.video:
        run_video_or_camera(args.video, person_det, anomaly_det)
    elif args.camera:
        run_video_or_camera(0, person_det, anomaly_det)
    else:
        print("No input mode specified. Defaulting to demo on sample test image...")
        demo_img = os.path.join("dataset", "test", "normal", "norm_test_00000.jpg")
        if os.path.exists(demo_img):
            run_image(demo_img, person_det, anomaly_det, save_output=True)
        else:
            print("Usage: python app.py --image <path> | --video <path_or_cam_idx> | --camera")

if __name__ == "__main__":
    main()
