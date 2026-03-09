# -*- coding: utf-8 -*-
# PA-CFL Server: FedAvg aggregation within a single bubble
# One server instance is launched per bubble.
# Usage: python -m src.pacfl.server --bubble_id 0 --port 8090

import argparse
import numpy as np
import flwr as fl
import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--bubble_id", type=int, default=0)
parser.add_argument("--port", type=int, default=8090)
parser.add_argument("--min_clients", type=int, default=2)
parser.add_argument("--num_rounds", type=int, default=100)
args = parser.parse_args()

wandb.init(project="PA-CFL", config={"bubble_id": args.bubble_id})


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
    min_fit_clients=args.min_clients,
    min_evaluate_clients=args.min_clients,
    min_available_clients=args.min_clients,
    evaluate_metrics_aggregation_fn=weighted_average,
)

fl.server.start_server(
    server_address=f"0.0.0.0:{args.port}",
    config=fl.server.ServerConfig(num_rounds=args.num_rounds),
    strategy=strategy,
)
