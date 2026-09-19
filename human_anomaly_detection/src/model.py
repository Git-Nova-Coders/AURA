import torch
import torch.nn as nn
import torch.optim as optim
from . import config

class AnomalyCNN(nn.Module):
    """
    Baseline CNN for Human Anomaly Detection.
    Matches the architecture proposed in Milestone 3:
    Conv2D -> ReLU -> MaxPooling -> Conv2D -> ReLU -> MaxPooling -> Flatten/GlobalAveragePooling -> Dense -> Dropout -> Output
    """
    def __init__(self):
        super(AnomalyCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Input: (batch, 3, 224, 224) after permutation
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (batch, 32, 112, 112)
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (batch, 64, 56, 56)
        )
        
        # We use GlobalAveragePooling instead of Flatten to avoid a massive dense layer and overfitting
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1)) # (batch, 64, 1, 1)
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 1) # Single output for Binary Classification
        )
        
    def forward(self, x):
        # x arrives as (batch, H, W, C) from data_loader / OpenCV
        # PyTorch expects (batch, C, H, W).
        x = x.permute(0, 3, 1, 2)
        
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def get_loss_and_optimizer(model, learning_rate=0.001):
    """
    Returns the appropriate loss function and optimizer for training.
    """
    # BCEWithLogitsLoss combines Sigmoid and Binary Cross Entropy safely
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return criterion, optimizer

def calculate_metrics(outputs, labels):
    """
    Calculates basic batch metrics: accuracy.
    """
    predictions = (torch.sigmoid(outputs) >= 0.5).float()
    labels = labels.unsqueeze(1).float()
    
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    
    accuracy = correct / total
    return accuracy
