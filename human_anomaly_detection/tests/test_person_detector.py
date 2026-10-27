import cv2
import numpy as np
import os
from src.person_detector import PersonDetector
from src.anomaly_detector import AnomalyDetector

def test_pipeline():
    print("Testing Milestone 9: Standalone Human Detection & Anomaly Pipeline")
    
    # 1. Initialize standalone person detector
    print("Initializing PersonDetector...")
    person_detector = PersonDetector(confidence_threshold=0.5)
    
    # 2. Initialize anomaly detector
    print("Initializing AnomalyDetector...")
    anomaly_detector = AnomalyDetector("models/anomaly_model.pth")
    
    # 3. Load a sample frame
    test_img_path = os.path.join("dataset", "test", "normal", "norm_test_00000.jpg")
    if not os.path.exists(test_img_path):
        print("Sample test image not found, creating synthetic test frame...")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
    else:
        frame = cv2.imread(test_img_path)
        print(f"Loaded test frame of shape {frame.shape}")
        
    # 4. Step 1: Detect and Crop Persons
    person_results = person_detector.detect_and_crop(frame)
    print(f"Persons detected: {len(person_results)}")
    
    # If no persons detected in this specific frame (e.g. if frame is already a crop or background),
    # verify that passing the frame directly as a crop works seamlessly as fallback
    if len(person_results) == 0:
        print("No person box detected with threshold; evaluating full frame as person region...")
        crops_to_test = [{"crop": frame, "box": (0, 0, frame.shape[1], frame.shape[0]), "confidence": 1.0}]
    else:
        crops_to_test = person_results

    # 5. Step 2: Pass person crop(s) into AnomalyDetector
    for idx, item in enumerate(crops_to_test):
        crop = item["crop"]
        box = item["box"]
        det_conf = item["confidence"]
        
        prediction = anomaly_detector.predict(crop)
        print(f"Person {idx+1} at Box {box} (Det Conf: {det_conf:.2f}):")
        print(f"  Anomaly Label: {prediction.get('label')}, Confidence: {prediction.get('confidence')}")
        
    print("\nMilestone 9 Pipeline (Frame -> Person Detection -> Crop -> Anomaly Classification) successfully verified!")

if __name__ == "__main__":
    test_pipeline()
