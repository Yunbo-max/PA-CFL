# -*- coding: utf-8 -*-
# MLP model used in FedAvg, FedProx, IFCA, and local learning baselines.

import torch.nn as nn


class Net(nn.Module):
    def __init__(self, input_neurons, output_neurons=1, hidden_layers=4,
                 neurons_per_layer=64, dropout=0.3):
        super().__init__()
        self.input_neurons = input_neurons

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_neurons, neurons_per_layer))
        self.layers.append(nn.ReLU())

        for _ in range(hidden_layers):
            self.layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(p=dropout))

        self.layers.append(nn.Linear(neurons_per_layer, output_neurons))

    def forward(self, x):
        x = x.view(-1, self.input_neurons)
        for layer in self.layers:
            x = layer(x)
        return x
