# -*- coding: utf-8 -*-
# Transformer model for sales prediction.

import torch.nn as nn


class SalesPredictionTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1,
                 num_layers=2, num_heads=8, dropout=0.5):
        super().__init__()
        self.output_dim = output_dim

        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                hidden_dim, num_heads,
                dim_feedforward=hidden_dim, dropout=dropout),
            num_layers
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)
        x = self.fc(x)
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.output_dim)
        return x
