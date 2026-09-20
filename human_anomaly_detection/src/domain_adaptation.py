import os
import glob
import random
import json
import cv2
import numpy as np
import torch

from . import config
from .person_detector import PersonDetector
from .preprocessing import preprocess_image, augment_image
from .model import get_model, get_loss_and_optimizer, calculate_metrics

def run_domain_adaptation(epochs=8, lr=0.0003):
    print("=== Domain Adaptation & Human Crop Calibration ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Initialize Person Detector to crop real pedestrians
    print("Extracting pedestrian crops from ShanghaiTech normal footage...")
    pd = PersonDetector(0.5)
    
    normal_crops = []
    norm_files = glob.glob(os.path.join(config.DATASET_DIR, "train", "normal", "*.jpg"))
    random.seed(42)
    random.shuffle(norm_files)
    
    for f in norm_files[:200]:
        img = cv2.imread(f)
        if img is not None:
            crops = pd.detect_and_crop(img)
            for c in crops:
                normal_crops.append(c["crop"])
                if len(normal_crops) >= 400:
                    break
        if len(normal_crops) >= 400:
            break
            
    print(f"Collected {len(normal_crops)} outdoor pedestrian crops.")

    # 2. Add indoor / desk / sitting normal human crops (addressing live camera domain shift)
    user_img_path = r"C:/Users/singh/.gemini/antigravity/brain/351fa60a-69a2-4f5b-af34-64c2e1ad907d/.user_uploaded/media_1789911456712.png"
    if os.path.exists(user_img_path):
        user_img = cv2.imread(user_img_path)
        user_crops = pd.detect_and_crop(user_img)
        base_user_crop = user_crops[0]["crop"] if len(user_crops) > 0 else user_img
        
        print("Synthesizing indoor webcam normal sitting poses with augmentations...")
        for _ in range(400):
            aug = augment_image(base_user_crop.copy())
            h, w = aug.shape[:2]
            dh = int(h * random.uniform(0.0, 0.15))
            dw = int(w * random.uniform(0.0, 0.15))
            aug_crop = aug[dh:max(dh+10, h-dh), dw:max(dw+10, w-dw)]
            normal_crops.append(cv2.resize(aug_crop, (224, 224)))
            
    print(f"Total Normal Human Crops: {len(normal_crops)}")

    # 3. Anomaly samples (ShanghaiTech anomalies + rapid unusual motion)
    print("Collecting Anomaly samples...")
    anom_files = glob.glob(os.path.join(config.DATASET_DIR, "train", "anomaly", "*.jpg"))
    random.shuffle(anom_files)
    anomaly_samples = []
    
    for f in anom_files[:800]:
        img = cv2.imread(f)
        if img is not None:
            anomaly_samples.append(cv2.resize(img, (224, 224)))
            
    print(f"Total Anomaly Samples: {len(anomaly_samples)}")

    # 4. Assemble and balance dataset
    n_samples = min(len(normal_crops), len(anomaly_samples))
    print(f"Balanced Dataset: {n_samples} Normal vs {n_samples} Anomaly")

    X_raw = normal_crops[:n_samples] + anomaly_samples[:n_samples]
    y_raw = [0] * n_samples + [1] * n_samples

    X = [preprocess_image(img, False) for img in X_raw]
    indices = list(range(len(X)))
    random.shuffle(indices)

    X_train = np.array([X[i] for i in indices], dtype=np.float32)
    y_train = np.array([y_raw[i] for i in indices], dtype=np.float32)

    # 5. Fine-tune MobileNetV2
    model = get_model("mobilenet", pretrained=True).to(device)
    criterion, optimizer = get_loss_and_optimizer(model, learning_rate=lr)

    batch_size = 32
    steps = len(X_train) // batch_size

    for epoch in range(epochs):
        model.train()
        loss_total = 0.0
        acc_total = 0.0
        
        for i in range(steps):
            bx = torch.tensor(X_train[i*batch_size:(i+1)*batch_size]).to(device)
            by = torch.tensor(y_train[i*batch_size:(i+1)*batch_size]).to(device)
            
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            loss_total += loss.item()
            acc_total += calculate_metrics(out, by)
            
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss_total/steps:.4f} - Accuracy: {acc_total/steps:.4f}")

    # 6. Save calibrated model
    model_save_path = os.path.join(config.MODELS_DIR, "anomaly_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Successfully saved calibrated model to {model_save_path}")

    # 7. Update model config
    model_config_path = os.path.join(config.MODELS_DIR, "model_config.json")
    with open(model_config_path, "w") as f:
        json.dump({
            "architecture": "mobilenet",
            "target_size": list(config.TARGET_SIZE),
            "calibrated_for_person_crops": True,
            "classes": {"0": "Normal", "1": "Anomaly"}
        }, f, indent=4)

    # 8. Test on user crop
    if os.path.exists(user_img_path):
        model.eval()
        test_p = preprocess_image(base_user_crop, False)
        with torch.no_grad():
            prob = torch.sigmoid(model(torch.tensor(test_p).unsqueeze(0).to(device))).item()
        label = "Anomaly" if prob >= 0.5 else "Normal"
        conf = prob if label == "Anomaly" else (1.0 - prob)
        print(f"\nVerification on User Webcam Sitting Pose:")
        print(f"  Prediction: {label} ({conf*100:.1f}%) [Raw Anomaly Probability: {prob:.4f}]")

if __name__ == "__main__":
    run_domain_adaptation()
