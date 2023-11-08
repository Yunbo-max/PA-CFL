# -*- coding = utf-8 -*-
# @time:04/08/2023 10:58
# Author:Yunbo Long
# @File:similarity_calculation.py
# @Software:PyCharm
import warnings
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import sigmoid_kernel


# Hiding the warnings
warnings.filterwarnings("ignore")
import wandb

import h5py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn import svm, metrics, tree, preprocessing, linear_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import (
    Ridge,
    LinearRegression,
    LogisticRegression,
    ElasticNet,
    Lasso,
)
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    BaggingClassifier,
    ExtraTreesClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    recall_score,
    confusion_matrix,
    f1_score,
    roc_curve,
    auc,
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from tensorboardX import SummaryWriter
from sklearn.model_selection import train_test_split
import warnings
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

torch.manual_seed(42)  # Set the random seed to 42 (you can use any integer)

# Hiding the warnings
warnings.filterwarnings("ignore")


# Define a custom dataset class
class MarketDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_data = self.inputs[idx]
        target = self.targets[idx]
        return input_data, target


import h5py
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# Set the number of clients, rounds, and epochs
# sheet_name = ['0', '1', '2']
sheet_name = [
    "0",
    "1",
    "2",
    "3",
    "5",
    "6",
    "7",
    "9",
    "10",
    "12",
    "14",
    "16",
    "17",
    "22",
]
num_round = [2, 3, 4]
num_epochs = 10

region_map = {
    0: "Southeast Asia",
    1: "South Asia",
    2: "Oceania",
    3: "Eastern Asia",
    4: "West Asia",
    5: "West of USA",
    6: "US Center",
    7: "West Africa",
    8: "Central Africa",
    9: "North Africa",
    10: "Western Europe",
    11: "Northern Europe",
    12: "Central America",
    13: "Caribbean",
    14: "South America",
    15: "East Africa",
    16: "Southern Europe",
    17: "East of USA",
    18: "Canada",
    19: "Southern Africa",
    20: "Central Asia",
    21: "Eastern Europe",
    22: "South of USA",
}


# Define the neural network model
class Net(nn.Module):
    def __init__(
        self, input_neurons, output_neurons, hidden_layers, neurons_per_layer, dropout
    ):
        super(Net, self).__init__()

        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.dropout = dropout

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


# Set the parameters for the model
input_neurons = 25
output_neurons = 1
hidden_layers = 4
neurons_per_layer = 64
dropout = 0.3

# Initialize a shared global model
global_model = Net(
    input_neurons, output_neurons, hidden_layers, neurons_per_layer, dropout
)

# Open the HDF5 file
file = h5py.File(
    "market_data.h5",
    "r",
)

# Get the number of clients from sheet_name
num_clients = len(sheet_name)


# Set the number of iterations for federated learning
num_iterations = 10

# Initialize an empty similarity matrix to store similarity values for each pair of clients
similarity_matrix_total1 = np.zeros((len(sheet_name), len(sheet_name)))
similarity_matrix_total2 = np.zeros((len(sheet_name), len(sheet_name)))
similarity_matrix_total3 = np.zeros((len(sheet_name), len(sheet_name)))

# Initialize a dictionary to store metrics for each client and each iteration
# Initialize a dictionary to store metrics for each client and each iteration
# Initialize a dictionary to store metrics for each client, round, and iteration
# Initialize an empty similarity matrix to store similarity values for each pair of clients for each iteration
similarity_matrix_total = np.zeros((len(sheet_name), len(sheet_name), num_iterations))


for num_rounds in num_round:
    print(f"testing {num_rounds }")
    # Perform federated learning

    # Initialize a dictionary to store metrics for each client, round, and iteration
    metrics = {
        client: {"r2": [[[] for _ in range(num_rounds)] for _ in range(num_iterations)]}
        for client in sheet_name
    }

    # Initialize a list to store the feature matrices for each iteration
    all_feature_matrices = []

    # Initialize a list to store the similarity matrices for each iteration
    similarity_matrices = []

    for iteration in range(num_iterations):
        print(f"Iteration {iteration + 1}/{num_iterations}")

        # Initialize a shared global model
        global_model = Net(
            input_neurons, output_neurons, hidden_layers, neurons_per_layer, dropout
        )

        # Initialize an empty list to store the client models for this round
        client_models = []

        for round in range(num_rounds):
            print(f"Round {round + 1}/{num_rounds}")

            for client in sheet_name:
                # Load the state dict of the global model to the client model
                model = Net(
                    input_neurons,
                    output_neurons,
                    hidden_layers,
                    neurons_per_layer,
                    dropout,
                )
                model.load_state_dict(global_model.state_dict())

                dataset = file[client][:]
                dataset = pd.DataFrame(dataset)

                # Read the column names from the attributes
                column_names = file[client].attrs["columns"]

                # Assign column names to the dataset
                dataset.columns = column_names

                dataset = dataset.drop(columns=["Region Index"])

                # Preprocess the data
                train_data = dataset
                xs = train_data.drop(["Sales"], axis=1)
                ys = train_data["Sales"]
                xs_train, xs_test, ys_train, ys_test = train_test_split(
                    xs, ys, test_size=0.3, random_state=42
                )

                # Split training set into training and validation sets
                xs_train, xs_val, ys_train, ys_val = train_test_split(
                    xs_train, ys_train, test_size=0.2, random_state=42
                )

                # Convert data to tensors
                train_inputs = torch.tensor(xs_train.values, dtype=torch.float32)
                train_targets = torch.tensor(ys_train.values, dtype=torch.float32)
                val_inputs = torch.tensor(xs_val.values, dtype=torch.float32)
                val_targets = torch.tensor(ys_val.values, dtype=torch.float32)
                test_inputs = torch.tensor(xs_test.values, dtype=torch.float32)
                test_targets = torch.tensor(ys_test.values, dtype=torch.float32)

                # Create data loaders
                train_dataset = MarketDataset(train_inputs, train_targets)
                val_dataset = MarketDataset(val_inputs, val_targets)
                test_dataset = MarketDataset(test_inputs, test_targets)
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=32)
                test_loader = DataLoader(test_dataset, batch_size=32)

                # Define the neural network model
                class Net(nn.Module):
                    def __init__(
                        self,
                        input_neurons,
                        output_neurons,
                        hidden_layers,
                        neurons_per_layer,
                        dropout,
                    ):
                        super(Net, self).__init__()

                        self.input_neurons = input_neurons
                        self.output_neurons = output_neurons
                        self.hidden_layers = hidden_layers
                        self.neurons_per_layer = neurons_per_layer
                        self.dropout = dropout

                        self.layers = nn.ModuleList()
                        self.layers.append(nn.Linear(input_neurons, neurons_per_layer))
                        self.layers.append(nn.ReLU())

                        for _ in range(hidden_layers):
                            self.layers.append(
                                nn.Linear(neurons_per_layer, neurons_per_layer)
                            )
                            self.layers.append(nn.ReLU())
                            self.layers.append(nn.Dropout(p=dropout))

                        self.layers.append(nn.Linear(neurons_per_layer, output_neurons))

                    def forward(self, x):
                        x = x.view(-1, self.input_neurons)
                        for layer in self.layers:
                            x = layer(x)
                        return x

                input_neurons = train_inputs.shape[1]
                output_neurons = 1
                hidden_layers = 4
                neurons_per_layer = 64
                dropout = 0.3
                model = Net(
                    input_neurons,
                    output_neurons,
                    hidden_layers,
                    neurons_per_layer,
                    dropout,
                )

                criterion = nn.MSELoss()
                learning_rate = 0.005
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                train_losses = []
                val_losses = []
                train_rmse_list = []
                val_rmse_list = []
                mae_train_list = []
                rmse_train_list = []
                mape_train_list = []
                mse_train_list = []
                r2_train_list = []
                mae_val_list = []
                rmse_val_list = []
                mape_val_list = []
                mse_val_list = []
                r2_val_list = []
                mae_test_list = []
                rmse_test_list = []
                mape_test_list = []
                mse_test_list = []
                r2_test_list = []

                for epoch in range(num_epochs):
                    train_losses = []
                    model.train()

                    for inputs, targets in train_loader:
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, targets.unsqueeze(1))
                        loss.backward()
                        optimizer.step()
                        train_losses.append(loss.item())

                    epoch_loss = np.mean(train_losses)

                # Use model to generate predictions for the test dataset
                client_models.append(model.state_dict())

                # Use model to generate predictions for the test dataset
                model.eval()
                with torch.no_grad():
                    test_preds = model(test_inputs)

                r2 = r2_score(test_targets.numpy(), test_preds.numpy())

                # Save the R2 value for the current round and iteration
                # Save the R2 value for the current round and iteration
                metrics[client]["r2"][iteration][round] = r2

            # Average the weights across all clients after each round
            averaged_weights = {
                k: sum(d[k] for d in client_models) / num_clients
                for k in client_models[0].keys()
            }

            # Update the global model
            global_model.load_state_dict(averaged_weights)

        # Create the feature matrix for the current iteration and all rounds
        feature_matrix = np.array(
            [
                [metrics[client]["r2"][iteration][r] for r in range(num_rounds)]
                for client in sheet_name
            ]
        )

        # print(feature_matrix)

        # Check if the feature matrix is empty (no valid R2 values)
        if feature_matrix.size == 0:
            print("No valid data in the feature matrix. Skipping this iteration.")
            continue

        # Append the feature matrix to the list after adding an additional dimension
        all_feature_matrices.append(np.expand_dims(feature_matrix, axis=2))

        # Concatenate the feature matrices along the third dimension to have shape (num_clients, num_rounds, num_iterations)
        feature_matrix_total = np.concatenate(all_feature_matrices, axis=2)

        # Step 2: Standardize the Data
        scaler = StandardScaler()

        # Flatten the last two dimensions
        flattened_data = feature_matrix_total.reshape(feature_matrix_total.shape[0], -1)
        normalized_data = scaler.fit_transform(flattened_data)

        # Compute Pairwise Similarity using Sigmoid Kernel for the current iteration
        similarity_matrix_total = sigmoid_kernel(normalized_data)
        # print(similarity_matrix_total)

        # Append the similarity matrix to the list of similarity matrices
        similarity_matrices.append(similarity_matrix_total)

    import numpy as np

    # Assuming similarity_matrices is a list of similarity matrices
    # Calculate the variance for each (i, j) position across similarity matrices
    variances = np.var(similarity_matrices, axis=0)

    # Calculate the mean for each (i, j) position across similarity matrices
    means = np.mean(similarity_matrices, axis=0)

    # Calculate the standard deviation for each (i, j) position across similarity matrices
    std_devs = np.std(similarity_matrices, axis=0)

    if num_rounds == 2:
        # Create dataframes for variances, means, and standard deviations
        variances_df = pd.DataFrame(variances)
        means_df = pd.DataFrame(means)
        std_devs_df = pd.DataFrame(std_devs)
    else:
        new_variances_df = pd.DataFrame(variances)
        new_means_df = pd.DataFrame(means)
        new_std_devs_df = pd.DataFrame(
            std_devs
        )  # Create dataframes for variances, means, and standard deviations
        # Concatenate the new row with the existing dataframe
        variances_df = pd.concat([variances_df, new_variances_df], ignore_index=True)
        means_df = pd.concat([means_df, new_means_df], ignore_index=True)
        std_devs_df = pd.concat([std_devs_df, new_std_devs_df], ignore_index=True)


print(variances_df)

print(means_df)

print(std_devs_df)

# Save variances_df to a CSV file
variances_df.to_csv("variances1-3.csv", index=False)

# Save means_df to a CSV file
means_df.to_csv("means1-3.csv", index=False)

# Save std_devs_df to a CSV file
std_devs_df.to_csv("std_devs1-3.csv", index=False)


# # Calculate other statistical values as needed (e.g., max, min, median, etc.)
# # For example, to calculate the maximum value for each (i, j) position:
# max_values = np.max(similarity_matrices, axis=0)

# # To calculate the minimum value for each (i, j) position:
# min_values = np.min(similarity_matrices, axis=0)

# # To calculate the median for each (i, j) position:
# medians = np.median(similarity_matrices, axis=0)

# You can use these computed values (variances, means, std_devs, max_values, min_values, medians) as needed for your analysis.

# import numpy as np
# import matplotlib.pyplot as plt

# Assuming you have computed the variances, means, std_devs, max_values, min_values, and medians as shown in the previous response

# # Define a function to create a heatmap for a given statistic
# def create_heatmap(data, title):
#     plt.figure(figsize=(10, 8))
#     plt.imshow(data, cmap='viridis', aspect='auto')
#     plt.title(title)
#     plt.colorbar()
#     plt.show()

# # Create heatmaps for variances, means, standard deviations, max values, min values, and medians
# create_heatmap(variances, 'Variances')
# create_heatmap(means, 'Means')
# create_heatmap(std_devs, 'Standard Deviations')
# create_heatmap(max_values, 'Maximum Values')
# create_heatmap(min_values, 'Minimum Values')
# create_heatmap(medians, 'Medians')


# # Save the similarity matrices for each iteration
# for iteration, similarity_matrix in enumerate(similarity_matrices):
#     np.save(f'similarity_matrix_total{iteration + 1}_3rounds.npy', similarity_matrix)
#     print('The {iteration} similarity_matrix is',iteration,similarity_matrix)

# # Save the similarity matrices for each iteration
# for iteration, similarity_matrix in enumerate(similarity_matrices):
#     np.save(f'similarity_matrix_total{iteration + 1}_5rounds.npy', similarity_matrix)
#     print('The {iteration} similarity_matrix is',iteration,similarity_matrix)

# # Save the similarity matrices for each iteration
# for iteration, similarity_matrix in enumerate(similarity_matrices):
#     np.save(f'similarity_matrix_total{iteration + 1}_10rounds.npy', similarity_matrix)
#     print('The {iteration} similarity_matrix is',iteration,similarity_matrix)
