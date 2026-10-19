"""
AURA HAD Deep Learning Model Architectures (PyTorch)
Contains:
1. MobileNetV2AnomalyClassifier: Pretrained MobileNetV2 backbone with regularized binary classification head
2. AnomalyCNN: Baseline 2-block convolutional neural network
3. Model factory and utility functions
"""

import os
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as tv_models


class MobileNetV2AnomalyClassifier(nn.Module):
    """
    ImageNet pretrained MobileNetV2 with inverted residuals & BatchNorm,
    coupled with a regularized binary classification head.
    Trained on ShanghaiTech and calibrated for real-time person crops.
    """
    def __init__(self, freeze_backbone: bool = False):
        super().__init__()
        # Load weights without downloading if possible, weights will be loaded from checkpoint
        backbone = tv_models.mobilenet_v2(weights=None)
        self.features = backbone.features

        if freeze_backbone:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accepts (B, H, W, C) from OpenCV or (B, C, H, W)
        if x.dim() == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class AnomalyCNN(nn.Module):
    """
    Baseline CNN for Human Anomaly Detection.
    Conv2D -> ReLU -> MaxPool -> Conv2D -> ReLU -> MaxPool -> AdaptiveAvgPool -> Dense -> Dropout -> Output
    """
    def __init__(self):
        super().__init__()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_model(architecture: str = "mobilenet", pretrained: bool = False) -> nn.Module:
    """Factory function to build baseline or improved model architectures."""
    if architecture.lower() in ("mobilenet", "mobilenetv2", "improved"):
        return MobileNetV2AnomalyClassifier(freeze_backbone=pretrained)
    return AnomalyCNN()
