"""
AURA Reliability ANN Model Definition
Implements the PyTorch MLP architecture specified in the System Software Design (SSD) document:
Dense(32) -> ReLU -> Dense(16) -> ReLU -> Dense(1) -> Sigmoid -> Reliability score
"""

import torch
import torch.nn as nn
from typing import Dict, Any


class ReliabilityANN(nn.Module):
    """
    Custom Multilayer Perceptron (MLP) for estimating object detection reliability.
    
    Architecture:
        Input (input_dim) -> Dense(32) -> ReLU -> Dense(16) -> ReLU -> Dense(1) -> Sigmoid
    """
    def __init__(self, input_dim: int = 14):
        super().__init__()
        self.input_dim = input_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs forward pass of the model.
        
        Args:
            x (torch.Tensor): Input features of shape (Batch, input_dim)
            
        Returns:
            torch.Tensor: Reliability scores in range [0.0, 1.0] of shape (Batch, 1)
        """
        return self.network(x)

    def get_metadata(self) -> Dict[str, Any]:
        """Returns architecture metadata for tracing/logging."""
        return {
            "model_class": self.__class__.__name__,
            "input_dim": self.input_dim,
            "architecture": "Dense(32) -> ReLU -> Dense(16) -> ReLU -> Dense(1) -> Sigmoid",
        }
