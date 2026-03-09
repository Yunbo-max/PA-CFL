# -*- coding: utf-8 -*-
# FedAvg Server
# Usage: python -m src.baselines.fedavg.server

import numpy as np
import flwr as fl
import wandb

wandb.init(project="PA-CFL_FedAvg")


def weighted_average(metrics):
    examples = [n for n, _ in metrics]
    total = sum(examples)
    result = {}
    for key in ["rmse", "r_squared", "mse", "mae", "loss"]:
        values = [n * m[key] for n, m in metrics if key in m]
        if values:
            result[key] = sum(values) / total
    wandb.log(result)
    return result


strategy = fl.server.strategy.FedAvg(
    min_fit_clients=9,
    min_evaluate_clients=9,
    min_available_clients=9,
    evaluate_metrics_aggregation_fn=weighted_average,
)

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=100),
    strategy=strategy,
)
