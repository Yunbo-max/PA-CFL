# -*- coding = utf-8 -*-
# @time:04/08/2023 10:58
# Author:Yunbo Long
# @File:similarity_calculation.py
# @Software:PyCharm
import warnings

warnings.filterwarnings("ignore")

import torch
import h5py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
import sys  # Import the sys module
import logging.config
import yaml
import os


def setup_logging():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(script_dir, "logging_config.yaml")
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    logging.config.dictConfig(config)


# Call the setup_logging function before any logging statements in your code
setup_logging()


from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics.pairwise import sigmoid_kernel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn import metrics
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

# Set the random seed
torch.manual_seed(42)

import ray


num_round = [i for i in range(3, 100)]
num_epochs = 5
num_iterations = 10


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


# Define the parallel function to process a single client
@ray.remote
def process_client(client, device, client_models):
    # Set the number of clients, rounds, and epochs
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

    client_id = int(client)
    # Open the HDF5 file inside the function
    file = h5py.File("Data/market_data.h5", "r")
    dataset = file[client][:]

    # Use client_id to access the correct dataset

    dataset = pd.DataFrame(dataset)

    # Read the column names from the attributes
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
    train_inputs = torch.tensor(xs_train.values, dtype=torch.float32).to(device)
    train_targets = torch.tensor(ys_train.values, dtype=torch.float32).to(device)
    val_inputs = torch.tensor(xs_val.values, dtype=torch.float32).to(device)
    val_targets = torch.tensor(ys_val.values, dtype=torch.float32).to(device)
    test_inputs = torch.tensor(xs_test.values, dtype=torch.float32).to(device)
    test_targets = torch.tensor(ys_test.values, dtype=torch.float32).to(device)

    # Create data loaders
    train_dataset = MarketDataset(train_inputs, train_targets)
    # val_dataset = MarketDataset(val_inputs, val_targets)
    test_dataset = MarketDataset(test_inputs, test_targets)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

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
    ).to(device)

    criterion = nn.MSELoss()
    learning_rate = 0.005
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []

    for epoch in range(num_epochs):
        train_losses = []
        model.train()

        for batch_inputs, batch_targets in train_loader:
            optimizer.zero_grad()
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Calculate metrics or print loss for the epoch if needed
        epoch_loss = np.mean(train_losses)

        # Print and flush the output
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        logger = logging.getLogger(__name__)
        logger.info(f"(Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # for epoch in range(num_epochs):
    #     train_losses = []
    #     model.train()

    #     # Inside the training loop:
    #     for inputs, targets in train_loader:
    #         optimizer.zero_grad()
    #         inputs = inputs.to(device)  # Move inputs to device
    #         targets = targets.to(device)  # Move targets to device
    #         outputs = model(inputs)
    #         loss = criterion(outputs, targets.unsqueeze(1))
    #         loss.backward()
    #         optimizer.step()
    #         train_losses.append(loss.item())

    # Use model to generate predictions for the test dataset
    model.eval()
    with torch.no_grad():
        test_inputs = test_inputs.to(device)  # Move test inputs to device
        test_preds = model(test_inputs)

    # Move tensors to the CPU and then convert to NumPy arrays
    # Calculate the R-squared (R2) score
    test_targets_np = test_targets.cpu().numpy()
    test_preds_np = test_preds.cpu().numpy()
    r2 = r2_score(test_targets_np, test_preds_np)

    # Log the R2 score for this client
    logger = logging.getLogger(__name__)
    logger.info(f"Client {client} - R2 Score: {r2}")

    client_models.append(model.state_dict())

    return r2, model.state_dict()


def main():
    # Initialize Ray
    ray.init(ignore_reinit_error=True, logging_level=logging.INFO)

    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)

    device = torch.device("cpu")
    print("Using Device: ", device)

    # Set the number of iterations,rounds,epochs for federated learning

    # Set the number of clients, rounds, and epochs
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

    # Set the parameters for the model
    input_neurons = 25
    output_neurons = 1
    hidden_layers = 4
    neurons_per_layer = 64
    dropout = 0.3

    # # Initialize a shared global model
    # global_model = Net(
    #     input_neurons, output_neurons, hidden_layers, neurons_per_layer, dropout
    # )

    # Get the number of clients from sheet_name
    num_clients = len(sheet_name)

    # # Initialize an empty similarity matrix to store similarity values for each pair of clients
    # similarity_matrix_total1 = np.zeros((len(sheet_name), len(sheet_name)))
    # similarity_matrix_total2 = np.zeros((len(sheet_name), len(sheet_name)))
    # similarity_matrix_total3 = np.zeros((len(sheet_name), len(sheet_name)))

    # Initialize an empty similarity matrix to store similarity values for each pair of clients for each iteration
    similarity_matrix_total = np.zeros(
        (len(sheet_name), len(sheet_name), num_iterations)
    )

    for num_rounds in num_round:
        print(f"testing {num_rounds }")
        # Perform federated learning

        all_r2_values = []

        # Initialize a dictionary to store metrics for each client, round, and iteration
        metrics = {
            client: {
                "r2": [[[] for _ in range(num_rounds)] for _ in range(num_iterations)]
            }
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
            ).to(device)

            # Define the model variable here
            model = Net(
                input_neurons, output_neurons, hidden_layers, neurons_per_layer, dropout
            ).to(device)

            # Initialize an empty list to store the client models for this round
            client_models = []

            futures = []

            for round in range(num_rounds):
                # Initialize an empty list to store the client models for this round
                client_models = []
                print(f"Round {round + 1}/{num_rounds}")

                # Initialize a list to store the Ray object references for each client
                futures = []

                # Submit tasks to process each client in parallel
                for client in sheet_name:
                    future = process_client.remote(client, device, client_models)
                    futures.append(future)

                # Retrieve the results from completed tasks
                results = ray.get(futures)

                # Process the results as needed
                for client, (r2, model_parameter) in zip(sheet_name, results):
                    print(f"Client {client} - R2 Score: {r2}")
                    client_models.append(model_parameter)

                    metrics[client]["r2"][iteration][round] = r2

                    # Load the state dict of the global model to the client model
                    model.load_state_dict(global_model.state_dict())

                # for client in sheet_name:
                #     # Load the state dict of the global model to the client model
                #     model.load_state_dict(global_model.state_dict())
                # print(client_models)
                if client_models == []:
                    continue

                else:
                    # Average the weights across all clients after each round
                    averaged_weights = {
                        k: sum(d[k] for d in client_models) / num_clients
                        for k in client_models[0].keys()
                    }

                # Update the global model
                global_model.load_state_dict(averaged_weights)

            # print(feature_matrix)

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
            flattened_data = feature_matrix_total.reshape(
                feature_matrix_total.shape[0], -1
            )
            normalized_data = scaler.fit_transform(flattened_data)

            # Compute Pairwise Similarity using Sigmoid Kernel for the current iteration
            similarity_matrix_total = sigmoid_kernel(normalized_data)
            # print(similarity_matrix_total)

            # Append the similarity matrix to the list of similarity matrices
            similarity_matrices.append(similarity_matrix_total)

        # Assuming similarity_matrices is a list of similarity matrices
        # Calculate the variance for each (i, j) position across similarity matrices
        variances = np.var(similarity_matrices, axis=0)

        # Calculate the mean for each (i, j) position across similarity matrices
        means = np.mean(similarity_matrices, axis=0)

        # Calculate the standard deviation for each (i, j) position across similarity matrices
        std_devs = np.std(similarity_matrices, axis=0)

        print(variances)
        print(type(variances))

        if num_rounds == 3:
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
            variances_df = pd.concat(
                [variances_df, new_variances_df], ignore_index=True
            )
            means_df = pd.concat([means_df, new_means_df], ignore_index=True)
            std_devs_df = pd.concat([std_devs_df, new_std_devs_df], ignore_index=True)

        # Log variances_df to the log file
        logger = logging.getLogger(__name__)
        logger.info("Variances Dataframe:")
        logger.info(variances_df.to_string())

        # Log means_df to the log file
        logger.info("Means Dataframe:")
        logger.info(means_df.to_string())

        # Log std_devs_df to the log file (assuming you have std_devs_df)
        logger.info("Standard Deviations Dataframe:")
        logger.info(std_devs_df.to_string())

    # Shutdown Ray when finished
    ray.shutdown()


if __name__ == "__main__":
    main()
