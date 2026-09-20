import os
import argparse
import cv2
import time

from .anomaly_detector import AnomalyDetector

def main():
    parser = argparse.ArgumentParser(description="Run Human Anomaly Detection on a video stream.")
    parser.add_argument("--video", type=str, default="0", help="Path to video file, or '0' for the default webcam.")
    args = parser.parse_args()

    # 1. Initialize the model exactly ONCE (Milestone 8 Performance Rule)
    print("Initializing inference engine...")
    try:
        detector = AnomalyDetector("models/anomaly_model.pth")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
        
    print("Model loaded successfully.")

    # 2. Parse video source (integer 0 for webcam, string for file)
    video_source = args.video
    if video_source.isdigit():
        video_source = int(video_source)
        
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source '{args.video}'")
        return
        
    print(f"Starting video inference on '{args.video}'...")
    print("Press 'q' to safely exit the video window.")

    # 3. Process frame loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or cannot read frame.")
            break
            
        start_time = time.time()
        
        # 4. Run Prediction
        result = detector.predict(frame)
        
        fps = 1.0 / (time.time() - start_time)
        
        # 5. Display label and confidence
        if "error" not in result:
            label = result["label"]
            confidence_pct = result["confidence"] * 100
            
            # Formatting
            color = (0, 255, 0) if label == "Normal" else (0, 0, 255)
            text = f"{label} ({confidence_pct:.1f}%) | FPS: {fps:.1f}"
            
            # Draw background box for readability
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
            box_coords = ((10, 10), (10 + text_size[0] + 10, 10 + text_size[1] + 20))
            cv2.rectangle(frame, box_coords[0], box_coords[1], (0, 0, 0), cv2.FILLED)
            cv2.putText(frame, text, (15, 30 + text_size[1] // 2), font, font_scale, color, thickness)
        else:
            cv2.putText(frame, "Inference Error", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 6. Display Video Window
        cv2.imshow("Human Anomaly Detection", frame)
        
        # 7. Safe exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exit requested by user.")
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
