import os
import argparse
import cv2

from .anomaly_detector import AnomalyDetector
from . import config

def main():
    parser = argparse.ArgumentParser(description="Run Human Anomaly Detection on a single image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image file.")
    parser.add_argument("--save", action="store_true", help="Save the annotated image to results/")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        return

    # 1. Initialize the inference engine
    try:
        detector = AnomalyDetector("models/anomaly_model.pth")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 2. Load the raw image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image from {args.image}. Ensure it is a valid image file.")
        return

    # 3. Run prediction
    result = detector.predict(image)

    if "error" in result:
        print(f"Prediction Error: {result['error']}")
        return

    label = result["label"]
    confidence_pct = result["confidence"] * 100

    # 4. Display result
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence_pct:.1f}%")

    # 5. Optionally save an annotated image
    if args.save:
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        
        # Color: Green for Normal, Red for Anomaly
        color = (0, 255, 0) if label == "Normal" else (0, 0, 255)
        text = f"{label} ({confidence_pct:.1f}%)"
        
        # Make a copy to annotate
        annotated = image.copy()
        
        # Add a filled rectangle for text background to ensure readability
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        box_coords = ((10, 10), (10 + text_size[0] + 10, 10 + text_size[1] + 20))
        cv2.rectangle(annotated, box_coords[0], box_coords[1], (0, 0, 0), cv2.FILLED)
        cv2.putText(annotated, text, (15, 30 + text_size[1] // 2), font, font_scale, color, thickness)
        
        # Save output
        filename = os.path.basename(args.image)
        output_path = os.path.join(config.RESULTS_DIR, f"annotated_{filename}")
        cv2.imwrite(output_path, annotated)
        print(f"Annotated image saved to: {output_path}")

if __name__ == "__main__":
    main()
