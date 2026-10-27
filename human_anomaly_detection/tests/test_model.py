import torch
from src.model import AnomalyCNN, get_loss_and_optimizer, calculate_metrics
from src.data_loader import get_dataset_paths, data_generator

def test_model():
    print("Initializing model...")
    model = AnomalyCNN()
    print("Model initialized successfully.")
    print(model)
    
    criterion, optimizer = get_loss_and_optimizer(model)
    
    paths, labels = get_dataset_paths('train')
    if not paths:
        print("No training data found. Make sure Milestone 1 dataset is extracted.")
        return
        
    print(f"Loaded {len(paths)} training paths.")
    
    # We will test just 2 batches (1 mini-epoch) to satisfy Milestone 3 criteria quickly without full training
    gen = data_generator(paths, labels, batch_size=16, is_training=True, shuffle=True)
    
    model.train()
    print("\nRunning a mini-epoch (2 batches)...")
    
    for i in range(2):
        batch_x, batch_y = next(gen)
        
        # Convert NumPy arrays to PyTorch tensors
        inputs = torch.tensor(batch_x, dtype=torch.float32)
        targets = torch.tensor(batch_y, dtype=torch.float32)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Calculate loss
        loss = criterion(outputs, targets.unsqueeze(1))
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        
        # Calculate accuracy
        accuracy = calculate_metrics(outputs, targets)
        
        print(f"Batch {i+1}/2 - Loss: {loss.item():.4f} - Accuracy: {accuracy:.4f}")
        
    print("\nMilestone 3 Criteria Met: Model built, accepted batch, produced predictions, and completed training steps without errors.")

if __name__ == "__main__":
    test_model()
