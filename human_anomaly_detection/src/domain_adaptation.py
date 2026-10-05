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
    print("=== Training Calibrated Human Anomaly Classifier ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pd = PersonDetector(0.5)
    
    # -------------------------------------------------------------
    # 1. NORMAL CLASS (Label: 0)
    #    - Upright walking pedestrians (ShanghaiTech normal)
    #    - Upright sitting/standing indoor webcam humans
    # -------------------------------------------------------------
    print("Gathering Normal Human samples...")
    normal_crops = []
    
    # A. Outdoor normal pedestrians from dataset
    norm_files = glob.glob(os.path.join(config.DATASET_DIR, "train", "normal", "*.jpg"))
    random.seed(42)
    random.shuffle(norm_files)
    
    for f in norm_files[:150]:
        img = cv2.imread(f)
        if img is not None:
            crops = pd.detect_and_crop(img)
            for c in crops:
                normal_crops.append(cv2.resize(c["crop"], (224, 224)))
                if len(normal_crops) >= 400:
                    break
        if len(normal_crops) >= 400:
            break
            
    print(f"  Outdoor pedestrian crops: {len(normal_crops)}")

    # B. Indoor webcam normal sitting poses with varied augmentations
    user_img_path = r"C:/Users/singh/.gemini/antigravity/brain/351fa60a-69a2-4f5b-af34-64c2e1ad907d/.user_uploaded/media_1789911456712.png"
    if os.path.exists(user_img_path):
        user_img = cv2.imread(user_img_path)
        user_crops = pd.detect_and_crop(user_img)
        base_user_crop = user_crops[0]["crop"] if len(user_crops) > 0 else user_img
        
        for _ in range(400):
            aug = augment_image(base_user_crop.copy())
            h, w = aug.shape[:2]
            dh = int(h * random.uniform(0.0, 0.15))
            dw = int(w * random.uniform(0.0, 0.15))
            aug_crop = aug[dh:max(dh+10, h-dh), dw:max(dw+10, w-dw)]
            normal_crops.append(cv2.resize(aug_crop, (224, 224)))
            
    print(f"  Total Normal Human Crops: {len(normal_crops)}")

    # -------------------------------------------------------------
    # 2. ANOMALY CLASS (Label: 1)
    #    - ShanghaiTech surveillance anomalies (vehicles, fights, chases)
    #    - Person fallen / collapsed horizontally (90 & 270 deg rotation)
    #    - Face covered / masked intruder
    # -------------------------------------------------------------
    print("Gathering Anomaly samples...")
    anomaly_samples = []
    
    # A. ShanghaiTech anomalous scenes
    anom_files = glob.glob(os.path.join(config.DATASET_DIR, "train", "anomaly", "*.jpg"))
    random.shuffle(anom_files)
    for f in anom_files[:400]:
        img = cv2.imread(f)
        if img is not None:
            anomaly_samples.append(cv2.resize(img, (224, 224)))
            
    print(f"  Dataset anomaly scenes: {len(anomaly_samples)}")

    # B. Simulated human falls & collapses (person slumped horizontally)
    if os.path.exists(user_img_path):
        for _ in range(200):
            rot_flag = random.choice([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE])
            fallen = cv2.rotate(base_user_crop, rot_flag)
            aug_fall = augment_image(fallen)
            anomaly_samples.append(cv2.resize(aug_fall, (224, 224)))
            
        # C. Covered face / masked intruder / hand distress
        for _ in range(200):
            masked = base_user_crop.copy()
            h, w = masked.shape[:2]
            y1, y2 = int(h * random.uniform(0.1, 0.2)), int(h * random.uniform(0.5, 0.7))
            x1, x2 = int(w * random.uniform(0.15, 0.3)), int(w * random.uniform(0.7, 0.85))
            masked[y1:y2, x1:x2] = np.random.randint(0, 40, (y2 - y1, x2 - x1, 3), dtype=np.uint8)
            anomaly_samples.append(cv2.resize(augment_image(masked), (224, 224)))
            
    print(f"  Total Anomaly Samples: {len(anomaly_samples)}")

    # -------------------------------------------------------------
    # 3. Balance and Train
    # -------------------------------------------------------------
    n_samples = min(len(normal_crops), len(anomaly_samples))
    print(f"Training on balanced set: {n_samples} Normal vs {n_samples} Anomaly")

    X_raw = normal_crops[:n_samples] + anomaly_samples[:n_samples]
    y_raw = [0] * n_samples + [1] * n_samples

    X = [preprocess_image(img, False) for img in X_raw]
    indices = list(range(len(X)))
    random.shuffle(indices)

    X_train = np.array([X[i] for i in indices], dtype=np.float32)
    y_train = np.array([y_raw[i] for i in indices], dtype=np.float32)

    model = get_model("mobilenet", pretrained=True).to(device)
    criterion, optimizer = get_loss_and_optimizer(model, learning_rate=lr)

    batch_size = 32
    steps = len(X_train) // batch_size

    for epoch in range(epochs):
        model.train()
        loss_total, acc_total = 0.0, 0.0
        
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

    # 4. Save model
    model_save_path = os.path.join(config.MODELS_DIR, "anomaly_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # 5. Save metadata
    model_config_path = os.path.join(config.MODELS_DIR, "model_config.json")
    with open(model_config_path, "w") as f:
        json.dump({
            "architecture": "mobilenet",
            "target_size": list(config.TARGET_SIZE),
            "calibrated_for_person_crops": True,
            "classes": {"0": "Normal", "1": "Anomaly"}
        }, f, indent=4)

if __name__ == "__main__":
    run_domain_adaptation()
