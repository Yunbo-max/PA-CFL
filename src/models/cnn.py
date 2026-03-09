# -*- coding: utf-8 -*-
# CNN model for sales prediction.

import torch
import torch.nn as nn


class SalesPredictionCNN(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, features)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = torch.mean(x, dim=2)  # Global Average Pooling
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
