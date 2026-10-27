import os
import cv2
import numpy as np
import torch

from src.anomaly_detector import AnomalyDetector
from src.person_detector import PersonDetector

def test_scenarios():
    print("=" * 65)
    print("      VERIFYING NORMAL VS ANOMALOUS HUMAN BEHAVIOR DETECTION")
    print("=" * 65)

    ad = AnomalyDetector("models/anomaly_model.pth")
    pd = PersonDetector(0.5)

    user_img_path = r"C:/Users/singh/.gemini/antigravity/brain/351fa60a-69a2-4f5b-af34-64c2e1ad907d/.user_uploaded/media_1789911456712.png"
    if not os.path.exists(user_img_path):
        print("User reference image not found.")
        return

    user_img = cv2.imread(user_img_path)
    crops = pd.detect_and_crop(user_img)
    base_user_crop = crops[0]["crop"] if len(crops) > 0 else user_img

    # -------------------------------------------------------------
    # Scenario 1: Normal Upright Seated Posture
    # -------------------------------------------------------------
    res1 = ad.predict(base_user_crop)
    prob1 = res1.get("anomaly_probability", 0.0) * 100
    print(f"\n[Scenario 1] Normal Upright Seated Pose (Webcam):")
    print(f"  -> Prediction: {res1['label']:<7} (Conf: {res1['confidence']*100:.1f}%) | Anomaly Prob: {prob1:.2f}%")
    print(f"  -> Expected:   Normal")

    # -------------------------------------------------------------
    # Scenario 2: Normal Walking Pedestrian Crop from Dataset
    # -------------------------------------------------------------
    ped_img = cv2.imread("dataset/test/normal/norm_test_00000.jpg")
    ped_crops = pd.detect_and_crop(ped_img)
    if len(ped_crops) > 0:
        res2 = ad.predict(ped_crops[0]["crop"])
        prob2 = res2.get("anomaly_probability", 0.0) * 100
        print(f"\n[Scenario 2] Normal Walking Pedestrian Crop (Dataset):")
        print(f"  -> Prediction: {res2['label']:<7} (Conf: {res2['confidence']*100:.1f}%) | Anomaly Prob: {prob2:.2f}%")
        print(f"  -> Expected:   Normal")

    # -------------------------------------------------------------
    # Scenario 3: Person Fallen / Collapsed (Rotated 90 degrees)
    # -------------------------------------------------------------
    fallen_crop = cv2.rotate(base_user_crop, cv2.ROTATE_90_CLOCKWISE)
    res3 = ad.predict(fallen_crop)
    prob3 = res3.get("anomaly_probability", 0.0) * 100
    print(f"\n[Scenario 3] Human Fall / Collapse (Horizontal Slump):")
    print(f"  -> Prediction: {res3['label']:<7} (Conf: {res3['confidence']*100:.1f}%) | Anomaly Prob: {prob3:.2f}%")
    print(f"  -> Expected:   Anomaly")

    # -------------------------------------------------------------
    # Scenario 4: Face Covered / Masked Intruder / Distress
    # -------------------------------------------------------------
    masked_crop = base_user_crop.copy()
    h, w = masked_crop.shape[:2]
    # Cover the entire face area
    masked_crop[int(h * 0.1):int(h * 0.65), int(w * 0.15):int(w * 0.85)] = 15
    res4 = ad.predict(masked_crop)
    prob4 = res4.get("anomaly_probability", 0.0) * 100
    print(f"\n[Scenario 4] Face Covered / Masked Intruder / Distress:")
    print(f"  -> Prediction: {res4['label']:<7} (Conf: {res4['confidence']*100:.1f}%) | Anomaly Prob: {prob4:.2f}%")
    print(f"  -> Expected:   Anomaly")

    # -------------------------------------------------------------
    # Scenario 5: Dataset Anomaly Scene (Bicycle / Vehicle in Walkway)
    # -------------------------------------------------------------
    anom_scene = cv2.imread("dataset/test/anomaly/anom_test_00000.jpg")
    res5 = ad.predict(cv2.resize(anom_scene, (224, 224)))
    prob5 = res5.get("anomaly_probability", 0.0) * 100
    print(f"\n[Scenario 5] Surveillance Anomaly Scene (Bicycle / Vehicle):")
    print(f"  -> Prediction: {res5['label']:<7} (Conf: {res5['confidence']*100:.1f}%) | Anomaly Prob: {prob5:.2f}%")
    print(f"  -> Expected:   Anomaly")

    print("\n" + "=" * 65)

if __name__ == "__main__":
    test_scenarios()
