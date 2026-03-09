# -*- coding: utf-8 -*-
# IFCA Client: Iterative Federated Clustering Algorithm
# Maintains K models, selects best cluster per round, trains only that model.
# Usage: python -m src.baselines.ifca.client --client_number 5 --config configs/ifca.json

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
parser.add_argument("--num_clusters", type=int, default=3)
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
testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

K = args.num_clusters
wandb.init(project=config["project_name"],
           config={"client_number": args.client_number, "region": region_name, "K": K})

input_dim = train_features.shape[1]
models = [Net(input_dim) for _ in range(K)]
optimizers = [optim.Adam(m.parameters(), lr=0.005) for m in models]
criterion = nn.MSELoss()


def get_all_params():
    params = []
    for m in models:
        params.extend([p.detach().numpy().astype('float32') for p in m.parameters()])
    return params


def set_all_params(parameters):
    n_per_model = len(list(models[0].parameters()))
    for k in range(K):
        start = k * n_per_model
        for param, new_param in zip(models[k].parameters(), parameters[start:start + n_per_model]):
            param.data.copy_(torch.from_numpy(new_param).float())


def select_best_cluster():
    best_k, best_loss = 0, float('inf')
    for k in range(K):
        models[k].eval()
        with torch.no_grad():
            loss = criterion(models[k](train_features).squeeze(), train_labels).item()
        if loss < best_loss:
            best_loss = loss
            best_k = k
    return best_k


class IFCAClient(fl.client.NumPyClient):
    def get_parameters(self, config=None):
        return get_all_params()

    def fit(self, parameters, config):
        set_all_params(parameters)
        best_k = select_best_cluster()
        wandb.log({"selected_cluster": best_k})

        m = models[best_k]
        opt = optimizers[best_k]

        start_time = time.time()
        for epoch in range(10):
            perm = torch.randperm(train_features.size(0))
            for i in range(0, train_features.size(0), 32):
                idx = perm[i:i+32]
                opt.zero_grad()
                loss = criterion(m(train_features[idx]), train_labels[idx].unsqueeze(1))
                loss.backward()
                opt.step()

        wandb.log({"Communication Time": time.time() - start_time})
        return get_all_params(), len(train_dataset), {"cluster_id": float(best_k)}

    def evaluate(self, parameters, config=None):
        set_all_params(parameters)
        best_k = select_best_cluster()

        predictions, targets, loss = evaluate_model(models[best_k], testloader)
        metrics = compute_metrics(targets, predictions)
        metrics["loss"] = loss
        wandb.log(metrics)
        return loss, len(test_dataset), metrics


fl.client.start_numpy_client(
    server_address=config["server_address"], client=IFCAClient())
