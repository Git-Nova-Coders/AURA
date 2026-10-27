import glob
import cv2
from src.anomaly_detector import AnomalyDetector

def main():
    ad = AnomalyDetector("models/anomaly_model.pth")

    norm_samples = glob.glob("dataset/test/normal/*.jpg")[:5]
    anom_samples = glob.glob("dataset/test/anomaly/*.jpg")[:5]

    print("==================================================")
    print("   VERIFYING ANOMALY DETECTOR ON UNSEEN TEST DATA")
    print("==================================================")

    print("\n--- 1. Testing Ground-Truth NORMAL Samples ---")
    for p in norm_samples:
        img = cv2.imread(p)
        res = ad.predict(img)
        prob = res.get("anomaly_probability", 0.0) * 100
        print(f"[{res['label']:<7}] Conf: {res['confidence']*100:>5.1f}% | AnomProb: {prob:>5.1f}% | File: {p}")

    print("\n--- 2. Testing Ground-Truth ANOMALY Samples ---")
    for p in anom_samples:
        img = cv2.imread(p)
        res = ad.predict(img)
        prob = res.get("anomaly_probability", 0.0) * 100
        print(f"[{res['label']:<7}] Conf: {res['confidence']*100:>5.1f}% | AnomProb: {prob:>5.1f}% | File: {p}")

    print("==================================================")

if __name__ == "__main__":
    main()
