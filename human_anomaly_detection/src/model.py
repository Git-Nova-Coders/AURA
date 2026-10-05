import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as tv_models
from . import config

class AnomalyCNN(nn.Module):
    """
    Baseline CNN for Human Anomaly Detection (Milestone 3).
    Conv2D -> ReLU -> MaxPool -> Conv2D -> ReLU -> MaxPool -> AdaptiveAvgPool -> Dense -> Dropout -> Output
    """
    def __init__(self):
        super(AnomalyCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        # Transpose OpenCV BGR/RGB input (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class MobileNetV2AnomalyClassifier(nn.Module):
    """
    Improved Architecture (Milestone 12 - Model Improvement).
    Uses ImageNet pretrained MobileNetV2 with inverted residuals & BatchNorm,
    coupled with a regularized binary classification head.
    """
    def __init__(self, freeze_backbone=True):
        super(MobileNetV2AnomalyClassifier, self).__init__()
        
        backbone = tv_models.mobilenet_v2(weights=tv_models.MobileNet_V2_Weights.DEFAULT)
        self.features = backbone.features
        
        if freeze_backbone:
            # Freeze early feature layers to preserve learned low-level representations
            for param in self.features.parameters():
                param.requires_grad = False
                
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # Permute (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_model(architecture="baseline", pretrained=True):
    """Factory function to build baseline or improved model architectures."""
    if architecture.lower() in ["mobilenet", "mobilenetv2", "improved"]:
        return MobileNetV2AnomalyClassifier(freeze_backbone=pretrained)
    return AnomalyCNN()


def get_loss_and_optimizer(model, learning_rate=0.001, pos_weight=None):
    """
    Returns loss function, optimizer, and optional class weighting.
    """
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    else:
        criterion = nn.BCEWithLogitsLoss()
        
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    return criterion, optimizer


def calculate_metrics(outputs, labels):
    """Calculates batch accuracy."""
    predictions = (torch.sigmoid(outputs) >= 0.5).float()
    labels = labels.unsqueeze(1).float()
    
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    
    return correct / total if total > 0 else 0.0
