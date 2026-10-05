import cv2
import json
from src.anomaly_detector import AnomalyDetector
from src.data_loader import get_dataset_paths

def test_inference_engine():
    print("Initializing AnomalyDetector...")
    detector = AnomalyDetector("models/anomaly_model.pth")
    print("Detector initialized successfully.")
    
    # Grab a sample test image
    paths, labels = get_dataset_paths('test')
    if not paths:
        print("No test data found.")
        return
        
    sample_img_path = paths[0]
    print(f"Testing inference on: {sample_img_path}")
    
    # Load raw BGR image using OpenCV
    raw_image = cv2.imread(sample_img_path)
    
    # Run prediction
    result = detector.predict(raw_image)
    
    print("\nInference Result:")
    print(json.dumps(result, indent=4))
    
    if "label" in result and "confidence" in result:
        print("\nMilestone 6 Criteria Met: A single image can be passed to the detector cleanly.")
    else:
        print("\nFailed to meet Milestone 6 criteria.")

if __name__ == "__main__":
    test_inference_engine()
