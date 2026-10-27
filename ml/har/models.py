"""
AURA HAR Deep Learning Model Architectures (PyTorch)
Contains:
1. SpatialPoseCNN (1D / 2D ConvNet over Pose Landmarks)
2. TemporalPoseLSTM (Multi-layer LSTM over Temporal Sequences)
3. HybridPoseCNNLSTM (Integrated ConvNet Spatial Feature Extractor + LSTM Temporal Classifier)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialPoseCNN(nn.Module):
    """
    CNN Architecture for Spatial Pose Sequence Classification.
    Uses 1D Convolutions over the feature channels across the temporal timeline,
    or 2D Convolutions across (Time, Landmarks).

    Input Shape: (Batch, Channels=132, Sequence=30)
    """

    def __init__(self, in_channels: int = 132, seq_len: int = 30, num_classes: int = 8, dropout: float = 0.3):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.num_classes = num_classes

        # Conv Block 1
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)

        # Conv Block 2
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(kernel_size=2)  # seq_len // 2 = 15

        # Conv Block 3
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(kernel_size=2)  # 15 // 2 = 7

        # Global Average Pooling + Fully Connected
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x input: (B, T=30, 33, 4) -> flatten to (B, T=30, 132) -> transpose to (B, C=132, T=30)
        if x.dim() == 4:
            B, T, L, D = x.shape
            x = x.view(B, T, L * D)
        if x.shape[1] == self.seq_len and x.shape[2] == self.in_channels:
            x = x.transpose(1, 2)  # (B, 132, 30)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))
        x = self.pool2(F.relu(self.bn3(self.conv3(x))))

        x = self.gap(x).squeeze(-1)  # (B, 256)
        x = self.dropout(F.relu(self.fc1(x)))
        logits = self.fc2(x)
        return logits


class TemporalPoseLSTM(nn.Module):
    """
    LSTM Architecture for Temporal Pose Dynamics.
    Input Shape: (Batch, Sequence=30, Features=132)
    """

    def __init__(
        self,
        in_features: int = 132,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 8,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * num_directions, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x input: (B, T, 33, 4) -> (B, T, 132)
        if x.dim() == 4:
            B, T, L, D = x.shape
            x = x.view(B, T, L * D)

        out, (hn, cn) = self.lstm(x)  # out: (B, T, hidden_size * num_directions)

        # Use attention or mean pooling over time
        temporal_mean = torch.mean(out, dim=1)  # (B, hidden_size * num_directions)
        x = self.dropout(F.relu(self.fc1(temporal_mean)))
        logits = self.fc2(x)
        return logits


class HybridPoseCNNLSTM(nn.Module):
    """
    Main Proposed Architecture: CNN + LSTM
    Pose Sequence (B, T, 33, 4)
         ↓
    TimeDistributed 1D CNN / ConvBlock (learns spatial representation at each timestep)
         ↓
    LSTM (learns temporal kinematics across timesteps)
         ↓
    Classification Head (8 Activity Classes)
    """

    def __init__(
        self,
        num_landmarks: int = 33,
        dim_per_lm: int = 4,
        seq_len: int = 30,
        cnn_out_dim: int = 128,
        lstm_hidden: int = 128,
        num_classes: int = 8,
        dropout: float = 0.3
    ):
        super().__init__()
        self.seq_len = seq_len
        self.in_features = num_landmarks * dim_per_lm  # 132

        # 1. Spatial Feature Extractor (per frame)
        self.spatial_cnn = nn.Sequential(
            nn.Linear(self.in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, cnn_out_dim),
            nn.BatchNorm1d(cnn_out_dim),
            nn.ReLU(),
        )

        # 2. Temporal Sequence Learner
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        # 3. Classifier Head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_hidden * 2, 64)
        self.bn_head = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x input: (B, T, 33, 4) -> (B, T, 132)
        if x.dim() == 4:
            B, T, L, D = x.shape
            x = x.view(B, T, L * D)
        else:
            B, T, _ = x.shape

        # TimeDistributed Spatial Representation
        # Reshape to (B * T, in_features)
        x_reshaped = x.view(B * T, -1)
        spatial_feats = self.spatial_cnn(x_reshaped)  # (B * T, cnn_out_dim)
        spatial_seq = spatial_feats.view(B, T, -1)     # (B, T, cnn_out_dim)

        # Temporal Modeling via LSTM
        lstm_out, _ = self.lstm(spatial_seq)  # (B, T, lstm_hidden * 2)

        # Temporal Pooling
        pooled = torch.mean(lstm_out, dim=1)  # (B, lstm_hidden * 2)

        # Head
        h = self.dropout(F.relu(self.bn_head(self.fc1(pooled))))
        logits = self.fc2(h)
        return logits
