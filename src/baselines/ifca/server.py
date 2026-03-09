# -*- coding: utf-8 -*-
# IFCA Server: maintains K cluster models, aggregates per-cluster
# Usage: python -m src.baselines.ifca.server

import numpy as np
import torch
import torch.nn as nn
import flwr as fl
import wandb

from typing import Dict, List, Optional, Tuple, Union
from flwr.common import (
    EvaluateIns, EvaluateRes, FitIns, FitRes, NDArrays,
    Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from flwr.server.strategy.aggregate import weighted_loss_avg
from src.models import Net

wandb.init(project="PA-CFL_IFCA")

NUM_CLUSTERS = 3
INPUT_DIM = 27  # Number of features after preprocessing


def make_initial_params():
    """Create K randomly initialized models and return concatenated params."""
    params = []
    for _ in range(NUM_CLUSTERS):
        m = Net(INPUT_DIM)
        params.extend([p.detach().numpy().astype('float32') for p in m.parameters()])
    return params


class IFCAStrategy(Strategy):
    def __init__(self):
        self.params_per_model = len(list(Net(INPUT_DIM).parameters()))

    def initialize_parameters(self, client_manager):
        return ndarrays_to_parameters(make_initial_params())

    def configure_fit(self, server_round, parameters, client_manager):
        fit_ins = FitIns(parameters, {"server_round": server_round})
        clients = client_manager.sample(num_clients=max(9, client_manager.num_available()),
                                         min_num_clients=9)
        return [(c, fit_ins) for c in clients]

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        cluster_results = {k: [] for k in range(NUM_CLUSTERS)}
        for _, fit_res in results:
            k = int(fit_res.metrics.get("cluster_id", 0))
            cluster_results[k].append(
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples))

        current = parameters_to_ndarrays(results[0][1].parameters)
        aggregated = list(current)

        for k in range(NUM_CLUSTERS):
            s = k * self.params_per_model
            e = s + self.params_per_model
            if cluster_results[k]:
                total = sum(n for _, n in cluster_results[k])
                avg = [np.zeros_like(p) for p in current[s:e]]
                for params, n in cluster_results[k]:
                    w = n / total
                    for j, p in enumerate(params[s:e]):
                        avg[j] += w * p
                for j in range(len(avg)):
                    aggregated[s + j] = avg[j]

        return ndarrays_to_parameters(aggregated), {}

    def configure_evaluate(self, server_round, parameters, client_manager):
        eval_ins = EvaluateIns(parameters, {})
        clients = client_manager.sample(num_clients=max(9, client_manager.num_available()),
                                         min_num_clients=9)
        return [(c, eval_ins) for c in clients]

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}
        loss = weighted_loss_avg([(r.num_examples, r.loss) for _, r in results])
        examples = [r.num_examples for _, r in results]
        total = sum(examples)
        result = {}
        for key in ["rmse", "r_squared", "mse", "mae", "loss"]:
            vals = [r.num_examples * r.metrics[key] for _, r in results if key in r.metrics]
            if vals:
                result[key] = sum(vals) / total
        wandb.log(result)
        return loss, result

    def evaluate(self, server_round, parameters):
        return None


fl.server.start_server(
    server_address="0.0.0.0:8082",
    config=fl.server.ServerConfig(num_rounds=100),
    strategy=IFCAStrategy(),
)
