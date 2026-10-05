import os
import json
import argparse
import time
import torch
import matplotlib.pyplot as plt
from . import config
from .model import get_model, get_loss_and_optimizer, calculate_metrics
from .data_loader import get_dataset_paths, data_generator

def plot_history(history, save_path):
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def log_experiment(record: dict):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    exp_file = os.path.join(config.RESULTS_DIR, "experiments.json")
    experiments = []
    if os.path.exists(exp_file):
        try:
            with open(exp_file, "r") as f:
                experiments = json.load(f)
        except Exception:
            experiments = []
            
    experiments.append(record)
    with open(exp_file, "w") as f:
        json.dump(experiments, f, indent=4)

def run_training_pipeline(model_arch="mobilenet", epochs=20, patience=5, lr=0.0005, pos_weight=None):
    print(f"Starting training pipeline with architecture: '{model_arch}'...")
    
    # 1. Ensure directories exist
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # 2. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 3. Initialize Model, Loss, Optimizer
    model = get_model(model_arch).to(device)
    criterion, optimizer = get_loss_and_optimizer(model, learning_rate=lr, pos_weight=pos_weight)
    
    # 4. Prepare Data Loaders
    train_paths, train_labels = get_dataset_paths('train')
    val_paths, val_labels = get_dataset_paths('validation')
    
    if not train_paths or not val_paths:
        raise ValueError("Dataset not found. Ensure Milestone 1 is completed.")
        
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    
    steps_per_epoch = max(1, len(train_paths) // config.BATCH_SIZE)
    val_steps = max(1, len(val_paths) // config.BATCH_SIZE)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    epochs_no_improve = 0
    best_model_path = os.path.join(config.MODELS_DIR, 'anomaly_model.pth')
    
    start_time = time.time()
    completed_epochs = 0
    
    # 5. Training Loop
    for epoch in range(epochs):
        completed_epochs += 1
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.train()
        
        train_loss = 0.0
        train_acc = 0.0
        
        train_gen = data_generator(train_paths, train_labels, batch_size=config.BATCH_SIZE, is_training=True, shuffle=True)
        
        for step in range(steps_per_epoch):
            try:
                batch_x, batch_y = next(train_gen)
            except StopIteration:
                break
                
            inputs = torch.tensor(batch_x, dtype=torch.float32).to(device)
            targets = torch.tensor(batch_y, dtype=torch.float32).to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += calculate_metrics(outputs, targets)
            
            if (step + 1) % 15 == 0 or step == steps_per_epoch - 1:
                print(f"  Step {step+1}/{steps_per_epoch} - Loss: {loss.item():.4f}")
                
        avg_train_loss = train_loss / steps_per_epoch
        avg_train_acc = train_acc / steps_per_epoch
        
        # 6. Validation Phase
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        
        val_gen = data_generator(val_paths, val_labels, batch_size=config.BATCH_SIZE, is_training=False, shuffle=False)
        
        with torch.no_grad():
            for step in range(val_steps):
                try:
                    batch_x, batch_y = next(val_gen)
                except StopIteration:
                    break
                    
                inputs = torch.tensor(batch_x, dtype=torch.float32).to(device)
                targets = torch.tensor(batch_y, dtype=torch.float32).to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets.unsqueeze(1))
                
                val_loss += loss.item()
                val_acc += calculate_metrics(outputs, targets)
                
        avg_val_loss = val_loss / val_steps
        avg_val_acc = val_acc / val_steps
        
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
        
        # 7. Model Checkpointing & Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = avg_val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            
            # Save model config metadata
            model_config_path = os.path.join(config.MODELS_DIR, "model_config.json")
            with open(model_config_path, "w") as f:
                json.dump({
                    "architecture": model_arch,
                    "target_size": list(config.TARGET_SIZE),
                    "best_val_loss": round(best_val_loss, 4),
                    "best_val_acc": round(best_val_acc, 4),
                    "classes": {"0": "Normal", "1": "Anomaly"}
                }, f, indent=4)
                
            print(f"--> Saved best model ({model_arch}) to {best_model_path}")
        else:
            epochs_no_improve += 1
            print(f"--> No improvement for {epochs_no_improve} epochs.")
            if epochs_no_improve >= patience:
                print("Early stopping triggered!")
                break
                
    # 8. Save Training History Plot
    plot_path = os.path.join(config.RESULTS_DIR, 'training_history.png')
    plot_history(history, plot_path)
    
    # 9. Track Experiment
    duration = time.time() - start_time
    experiment_id = f"EXP-{int(time.time())}"
    log_experiment({
        "experiment_id": experiment_id,
        "architecture": model_arch,
        "epochs_requested": epochs,
        "epochs_completed": completed_epochs,
        "batch_size": config.BATCH_SIZE,
        "learning_rate": lr,
        "best_val_loss": round(best_val_loss, 4),
        "best_val_acc": round(best_val_acc, 4),
        "training_duration_seconds": round(duration, 2)
    })
    
    print(f"\nTraining complete. Model & metadata saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Human Anomaly Detection Model")
    parser.add_argument("--model", type=str, default="mobilenet", choices=["baseline", "mobilenet"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    args = parser.parse_args()
    
    run_training_pipeline(model_arch=args.model, epochs=args.epochs, patience=args.patience, lr=args.lr)
