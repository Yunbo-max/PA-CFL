# -*- coding: utf-8 -*-
# GRU model for sales prediction.

import torch
import torch.nn as nn


class SalesPredictionGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1,
                 num_layers=2, dropout=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, features)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        out, _ = self.gru(x, h0)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)
