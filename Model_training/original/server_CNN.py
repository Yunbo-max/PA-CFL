# -*- coding: utf-8 -*-
# @Author: Yunbo
# @Date:   2024-02-12 13:58:32
# @Last Modified by:   Yunbo
# @Last Modified time: 2024-10-04 01:25:40
# -*- coding = utf-8 -*-
# @time:03/07/2023 09:32
# Author:Yunbo Long
# @File:server.py
# @Software:PyCharm
import flwr as fl
import argparse
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


from flwr.server.client_proxy import ClientProxy
import wandb

import flwr as fl
import torch

from collections import OrderedDict

from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from flwr.common import (
    FitRes,
    Parameters,
    ReconnectIns,
    Scalar,)

from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

wandb.init(project="Server_Cambridge_sales_MLP_FL_8.4east")

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.strategy import Strategy

import wandb

#
from typing import Callable, Union, List, Optional, Tuple, Dict
import numpy as np
import torch
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate


class FedCustom:
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
    ) -> None:
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients


    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Create custom configs
        n_clients = len(clients)
        half_clients = n_clients // 2
        standard_config = {"lr": 0.001}
        higher_lr_config = {"lr": 0.003}
        fit_configurations = []
        for idx, client in enumerate(clients):
            if idx < half_clients:
                fit_configurations.append((client, FitIns(parameters, standard_config)))
            else:
                fit_configurations.append(
                    (client, FitIns(parameters, higher_lr_config))
                )
        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average with respect to the number of examples used for training."""

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        # Calculate total number of examples
        total_examples = sum(num_examples for _, num_examples in weights_results)

        # Calculate weights based on the number of examples
        weights = [num_examples / total_examples for _, num_examples in weights_results]

        # Calculate weighted average of the parameters
        parameters_aggregated = ndarrays_to_parameters(
            aggregate(weights_results, aggregation_fn=self.custom_aggregation)
        )

        metrics_aggregated = {}
        return parameters_aggregated, metrics_aggregated

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []
        config = {}
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""

        if not results:
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        metrics_aggregated = {}
        return loss_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters using an evaluation function."""

        # Let's assume we won't perform the global model evaluation on the server side.
        return None

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def custom_aggregation(self, weights_results: List[Tuple[NDArrays, Scalar]]) -> NDArrays:
        """Custom aggregation function for weighted averaging."""
        aggregated_params = []
        for params, weight in weights_results:
            # Perform weighted averaging of the parameters
            weighted_params = [weight * param for param in params]
            aggregated_params.append(weighted_params)

        # Calculate the sum of the weighted parameters
        aggregated_params_sum = sum(aggregated_params)

        # Divide by the total number of clients for the final aggregation
        aggregated_params_avg = [param / len(weights_results) for param in aggregated_params_sum]

        return aggregated_params_avg



def weighted_average(metrics):
    losses = []
    rmse_values = []
    r_squared_values = []
    mse_values = []
    mae_values = []
    examples = []

    for num_examples, m in metrics:
        if "loss" in m:
            losses.append(num_examples * m["loss"])
        if "rmse" in m:
            rmse_values.append(num_examples * m["rmse"])
        if "r_squared" in m:
            r_squared_values.append(num_examples * m["r_squared"])
        if "mse" in m:
            mse_values.append(num_examples * m["mse"])
        if "mae" in m:
            mae_values.append(num_examples * m["mae"])
        examples.append(num_examples)

    loss = sum(losses) / sum(examples) if losses else None
    rmse = np.sqrt(sum(rmse_values) / sum(examples)) if rmse_values else None
    r_squared = sum(r_squared_values) / sum(examples) if r_squared_values else None
    mse = sum(mse_values) / sum(examples) if mse_values else None
    mae = sum(mae_values) / sum(examples) if mae_values else None

    metrics_dict = {
        "loss": loss,
        "rmse": rmse,
        "r_squared": r_squared,
        "mse": mse,
        "mae": mae,
    }

    wandb.log(metrics_dict)

    return metrics_dict




#
# fl.server.start_server(
#     server_address="0.0.0.0:8085",
#     config=fl.server.ServerConfig(num_rounds=5)
# )



# Create strategy
strategy = fl.server.strategy.FedAvg(
    # fraction_fit=0.5,
    # fraction_evaluate=0.5,
    min_fit_clients=9,
    min_evaluate_clients=9,
    min_available_clients=9,
    # evaluate_fn=get_evaluate_fn(model, args.toy),
    # on_fit_config_fn=fit_config,
    # on_evaluate_config_fn=evaluate_config,
    # initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
    evaluate_metrics_aggregation_fn=weighted_average,
)

# Start Flower server for four rounds of federated learning
fl.server.start_server(
    server_address="0.0.0.0:8002",
    config=fl.server.ServerConfig(num_rounds=100),
    strategy=strategy,
)


