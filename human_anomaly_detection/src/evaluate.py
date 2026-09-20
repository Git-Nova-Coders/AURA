import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

from . import config
from .model import AnomalyCNN
from .data_loader import get_dataset_paths, data_generator

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues, save_path=None):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save_path:
        plt.savefig(save_path)
    plt.close()

def evaluate_model():
    print("Starting evaluation pipeline on unseen TEST data...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load best model with dynamic architecture detection
    model_path = os.path.join(config.MODELS_DIR, 'anomaly_model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please run training first.")
        
    config_path = os.path.join(config.MODELS_DIR, "model_config.json")
    arch = "baseline"
    if os.path.exists(config_path):
        try:
            import json
            with open(config_path, "r") as f:
                arch = json.load(f).get("architecture", "baseline")
        except Exception:
            arch = "baseline"

    try:
        from .model import get_model
        model = get_model(arch, pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except Exception:
        model = AnomalyCNN()
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    model.to(device)
    model.eval()
    
    # 2. Load Test Dataset
    test_paths, test_labels = get_dataset_paths('test')
    if not test_paths:
        raise ValueError("Test dataset not found.")
        
    print(f"Evaluating on {len(test_paths)} test samples...")
    test_gen = data_generator(test_paths, test_labels, batch_size=config.BATCH_SIZE, is_training=False, shuffle=False)
    
    all_preds = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        while True:
            try:
                batch_x, batch_y = next(test_gen)
            except StopIteration:
                break
                
            inputs = torch.tensor(batch_x, dtype=torch.float32).to(device)
            outputs = model(inputs)
            
            probs = torch.sigmoid(outputs).squeeze(1).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(batch_y)
            
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # 3. Calculate Metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    # Compute Specificity and Sensitivity
    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0 # Identical to recall
    
    try:
        roc_auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        roc_auc = 0.0
        
    # 4. Save Classification Report
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(config.RESULTS_DIR, 'classification_report.txt')
    
    report_str = f"--- Anomaly Classification Report ---\n\n"
    report_str += f"Accuracy:    {accuracy:.4f}\n"
    report_str += f"Precision:   {precision:.4f}\n"
    report_str += f"Recall:      {recall:.4f} (Sensitivity)\n"
    report_str += f"Specificity: {specificity:.4f}\n"
    report_str += f"F1-score:    {f1:.4f}\n"
    report_str += f"ROC-AUC:     {roc_auc:.4f}\n\n"
    report_str += classification_report(all_targets, all_preds, target_names=["Normal", "Anomaly"], zero_division=0)
    
    with open(report_path, 'w') as f:
        f.write(report_str)
        
    print(f"\n{report_str}")
    print(f"--> Saved classification report to {report_path}")
    
    # 5. Save Confusion Matrix Plot
    cm = confusion_matrix(all_targets, all_preds)
    cm_path = os.path.join(config.RESULTS_DIR, 'confusion_matrix.png')
    plot_confusion_matrix(cm, classes=["Normal", "Anomaly"], save_path=cm_path)
    print(f"--> Saved confusion matrix plot to {cm_path}")

if __name__ == "__main__":
    evaluate_model()
