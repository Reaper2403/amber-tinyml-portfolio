"""Deployment-aware IMU multiclass model for Amber Stage 1."""

from __future__ import annotations

import torch
from torch import nn


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class Stage1ImuNet(nn.Module):
    """Input shape: [batch, time, channels] = [B, 150, 6]."""

    def __init__(self, num_classes: int = 7, in_channels: int = 6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.features = nn.Sequential(
            DepthwiseSeparableBlock(32, 64, kernel_size=7, stride=1),
            nn.MaxPool1d(kernel_size=2),
            DepthwiseSeparableBlock(64, 96, kernel_size=5, stride=1),
            nn.MaxPool1d(kernel_size=2),
            DepthwiseSeparableBlock(96, 128, kernel_size=5, stride=1),
            nn.MaxPool1d(kernel_size=2),
            DepthwiseSeparableBlock(128, 160, kernel_size=3, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(160, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return self.head(features)
