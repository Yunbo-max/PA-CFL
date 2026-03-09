# -*- coding: utf-8 -*-
# FedAvg Client
# Usage: python -m src.baselines.fedavg.client --client_number 5 --config configs/fedavg.json

import argparse
import json
import pickle
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import flwr as fl
import wandb

from src.data.dataset import RegressionDataset
from src.data.preprocess import REGION_MAP
from src.models import Net
from src.metrics import compute_metrics, evaluate_model

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--client_number", type=int, required=True)
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--data_path", type=str, default="data/processed/datasets.pkl")
args = parser.parse_args()

with open(args.config) as f:
    config = json.load(f)

with open(args.data_path, 'rb') as f:
    datasets = pickle.load(f)

region_name = REGION_MAP[args.client_number]
data = datasets[region_name]

train_features = data['train_features']
train_labels = data['train_labels']
test_features = data['test_features']
test_labels = data['test_labels']

train_dataset = RegressionDataset(train_features, train_labels)
test_dataset = RegressionDataset(test_features, test_labels)
trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

wandb.init(project=config["project_name"],
           config={"client_number": args.client_number, "region": region_name})

model = Net(train_features.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)


class FedAvgClient(fl.client.NumPyClient):
    def get_parameters(self, config=None):
        return [p.detach().numpy().astype('float32') for p in model.parameters()]

    def fit(self, parameters, config):
        for param, new_param in zip(model.parameters(), parameters):
            param.data.copy_(torch.from_numpy(new_param).float())

        start_time = time.time()
        for epoch in range(10):
            permutation = torch.randperm(train_features.size(0))
            batch_losses = []
            for i in range(0, train_features.size(0), 32):
                idx = permutation[i:i+32]
                optimizer.zero_grad()
                outputs = model(train_features[idx])
                loss = criterion(outputs, train_labels[idx].unsqueeze(1))
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())

        wandb.log({"Communication Time": time.time() - start_time})
        return self.get_parameters(), len(train_dataset), {}

    def evaluate(self, parameters, config=None):
        for param, new_param in zip(model.parameters(), parameters):
            param.data.copy_(torch.from_numpy(new_param).float())

        predictions, targets, loss = evaluate_model(model, testloader)
        metrics = compute_metrics(targets, predictions)
        metrics["loss"] = loss
        wandb.log(metrics)
        return loss, len(test_dataset), metrics


fl.client.start_numpy_client(
    server_address=config["server_address"], client=FedAvgClient())
