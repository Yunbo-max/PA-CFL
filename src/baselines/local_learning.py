# -*- coding: utf-8 -*-
# Unified local learning baseline.
# Usage: python -m src.baselines.local_learning --model {transformer,lstm,gru,cnn,mlp}
#
# Trains a local model per region (no federation) and logs to WandB.

import argparse
import pickle
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import wandb

from src.data.dataset import RegressionDataset
from src.data.preprocess import REGION_MAP, ACTIVE_REGIONS
from src.models import (
    Net, SalesPredictionTransformer, SalesPredictionLSTM,
    SalesPredictionGRU, SalesPredictionCNN,
)
from src.metrics import compute_metrics, evaluate_model

warnings.filterwarnings('ignore')

MODEL_REGISTRY = {
    'transformer': lambda dim: SalesPredictionTransformer(dim, hidden_dim=64, num_layers=2, num_heads=8, dropout=0.5),
    'lstm': lambda dim: SalesPredictionLSTM(dim, hidden_dim=64, num_layers=2, dropout=0.5),
    'gru': lambda dim: SalesPredictionGRU(dim, hidden_dim=64, num_layers=2, dropout=0.5),
    'cnn': lambda dim: SalesPredictionCNN(dim),
    'mlp': lambda dim: Net(dim, output_neurons=1, hidden_layers=4, neurons_per_layer=64, dropout=0.3),
}

LR_MAP = {
    'transformer': 0.00005,
    'lstm': 0.001,
    'gru': 0.001,
    'cnn': 0.001,
    'mlp': 0.005,
}


def train_local(model_name, data_path, num_epochs=100, batch_size=64):
    with open(data_path, 'rb') as f:
        datasets = pickle.load(f)

    for region_name, data in datasets.items():
        region_id = next((k for k, v in REGION_MAP.items() if v.strip() == region_name.strip()), None)
        if region_id is None or region_id not in ACTIVE_REGIONS:
            continue

        config = {"region": region_name, "model": model_name}
        with wandb.init(project=f'PA-CFL_{model_name}_local', config=config, reinit=True):

            train_features = data['train_features']
            train_labels = data['train_labels']
            test_features = data['test_features']
            test_labels = data['test_labels']
            input_dim = train_features.shape[1]

            train_dataset = RegressionDataset(train_features, train_labels)
            test_dataset = RegressionDataset(test_features, test_labels)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            model = MODEL_REGISTRY[model_name](input_dim)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=LR_MAP[model_name])

            print(f"\n{'='*60}")
            print(f"Region: {region_name} | Model: {model_name}")
            print(f"Train: {train_features.shape}, Test: {test_features.shape}")
            print(f"{'='*60}")

            for epoch in range(num_epochs):
                model.train()
                batch_losses = []
                for batch_inputs, batch_targets in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_inputs)
                    loss = criterion(outputs.squeeze(), batch_targets)
                    loss.backward()
                    optimizer.step()
                    batch_losses.append(loss.item())

                epoch_loss = np.mean(batch_losses)
                wandb.log({'Training Loss': epoch_loss})

                # Evaluate every epoch
                predictions, targets, test_loss = evaluate_model(model, test_loader)
                metrics = compute_metrics(targets, predictions)
                metrics['Test Loss'] = test_loss
                wandb.log(metrics)

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | "
                          f"RMSE: {metrics['rmse']:.4f} | R2: {metrics['r_squared']:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Local Learning Baseline')
    parser.add_argument('--model', type=str, required=True,
                        choices=['transformer', 'lstm', 'gru', 'cnn', 'mlp'])
    parser.add_argument('--data_path', type=str, default='data/processed/datasets.pkl')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    train_local(args.model, args.data_path, args.epochs, args.batch_size)
