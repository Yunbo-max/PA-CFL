# -*- coding: utf-8 -*-
# @Author: Yunbo
# @Date:   2024-02-01 20:19:10
# @Last Modified by:   Yunbo
# @Last Modified time: 2024-02-11 11:50:35

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
import pandas as pd
import networkx as nx
from sklearn.metrics import jaccard_score
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from grakel.kernels import GraphletSampling
from grakel.utils import graph_from_networkx
from sklearn.cluster import AgglomerativeClustering
import h5py
import xgboost as xgb
from scipy.stats import wasserstein_distance

# Step i: Reading Data and XGBoost Feature Importance

# Open the HDF5 file
file = h5py.File("/Users/yunbo/Documents/GitHub/PFL_Optimiozation/New_method/market_data.h5", "r")


# def setup_logging():
#     script_dir = os.path.dirname(os.path.realpath(__file__))
#     config_path = os.path.join(script_dir, "logging_config.yaml")
#     with open(config_path, "r") as config_file:
#         config = yaml.safe_load(config_file)
#     logging.config.dictConfig(config)


# # Call the setup_logging function before any logging statements in your code
# setup_logging()

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
sheet_names = ['0', '1', '2', '3', '5', '6', '7', '9', '10', '12', '14', '16', '17', '22']

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





# Initialize an empty list to store DataFrames from each sheet
dataframes = []

# Read and concatenate DataFrames from each sheet
for sheet_name in sheet_names:
    # Read the dataset using the current sheet name
    dataset = file[sheet_name][:]
    dataset = pd.DataFrame(dataset)

    # Read the column names from the attributes
    column_names = file[sheet_name].attrs["columns"]

    # Assign column names to the dataset
    dataset.columns = column_names

    # Append the DataFrame to the list
    dataframes.append(dataset)

# Concatenate all DataFrames into a single DataFrame
df = pd.concat(dataframes, ignore_index=True)

# Replace "Region Index" with "Order Region"
df.rename(columns={"Region Index": "Order Region"}, inplace=True)

# Replace numbers with corresponding names using region_map
df["Order Region"] = df["Order Region"].map(region_map)




# Function to calculate EMD between two distributions
def calculate_emd(distribution1, distribution2):
    return wasserstein_distance(distribution1, distribution2)

# List to store XGBoost feature importance scores for each region
feature_importance_scores = []
# Copy the original DataFrame to avoid modifying the original data
df_cluster = df.copy()

for region in df_cluster['Order Region'].unique():
    # Create subset DataFrame for the current Order Region
    subset_df = df_cluster[df_cluster['Order Region'] == region].drop(columns=['Order Region'])

    y = subset_df['Sales'].values

    X = subset_df.drop(columns=['Sales']).values

    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    model.fit(X, y)

    # Store feature importance scores
    feature_importance_scores.append(model.feature_importances_)

feature_importance_scores = np.array(feature_importance_scores)
# print(feature_importance_scores)

num_regions = len(feature_importance_scores)

similarity_matrix_xgboost = np.zeros((num_regions, num_regions))


# Reverse the label encoding to get the original region names
reverse_mapping = {v: k for k, v in region_map.items()}
# List to store region names
region_names = []
# Create t-SNE projections and store region names
# Create t-SNE projections and store region names
for i, region in enumerate(df_cluster['Order Region'].unique()):
    region_names.append(region_map.get(region, f' {region}'))





# Function to normalize feature importance scores
def normalize_feature_importance(feature_importance_scores):
    max_scores = np.max(feature_importance_scores, axis=1)
    normalized_scores = feature_importance_scores / max_scores[:, np.newaxis]
    return normalized_scores

# Normalize feature importance scores
normalized_feature_importance = normalize_feature_importance(feature_importance_scores)

# Calculate similarity matrix using Earth Mover's Distance
for i in range(num_regions):
    for j in range(i + 1, num_regions):
        emd_value = calculate_emd(normalized_feature_importance[i], normalized_feature_importance[j])
        similarity_matrix_xgboost[i, j] = emd_value
        similarity_matrix_xgboost[j, i] = emd_value

# Plot the similarity matrix
plt.figure(figsize=(30, 10))
plt.imshow(similarity_matrix_xgboost, cmap="YlGnBu", interpolation='nearest', aspect='auto')

# Add text annotations for each grid
for i in range(len(region_names)):
    for j in range(len(region_names)):
        plt.text(j, i, f'{similarity_matrix_xgboost[i, j]:.5f}', ha='center', va='center', color='black')




from itertools import combinations

# Generate all possible combinations of regions for each submatrix size
submatrix_combinations = {}
for submatrix_size in range(2, num_regions + 1):
    combinations_list = []
    for combination in combinations(region_names, submatrix_size):
        combinations_list.append(list(combination))
    submatrix_combinations[submatrix_size] = combinations_list

# Calculate the average similarity for each combination and submatrix size
average_similarities = {}
for submatrix_size, combinations in submatrix_combinations.items():
    similarities = []
    for combination in combinations:
        similarity_sum = 0
        count = 0
        for i in range(len(combination)):
            for j in range(i + 1, len(combination)):
                region_idx1 = region_names.index(combination[i])
                region_idx2 = region_names.index(combination[j])
                similarity_sum += similarity_matrix_xgboost[region_idx1, region_idx2]
                count += 1
        average_similarity = similarity_sum / count if count > 0 else 0
        similarities.append(average_similarity)
    average_similarities[submatrix_size] = similarities

# Sort submatrix sizes based on average similarity values
sorted_submatrix_sizes = sorted(average_similarities.keys(), key=lambda x: sum(average_similarities[x]))

# Create a list of tuples with combination and average similarity
combo_avg_similarity = []
for submatrix_size, combinations in submatrix_combinations.items():
    for idx, combination in enumerate(combinations, start=1):
        avg_similarity = average_similarities[submatrix_size][idx - 1]
        combo_avg_similarity.append((combination, avg_similarity))

# Sort the list of tuples by average similarity
sorted_combo_avg_similarity = sorted(combo_avg_similarity, key=lambda x: x[1])

# Initialize dictionary to store rankings for each region
region_rankings = {region: [] for region in region_names}

# Iterate through combinations and calculate rankings for each region
for combination, avg_similarity in sorted_combo_avg_similarity:
    for region in combination:
        region_rankings[region].append((combination, avg_similarity))







# Open the HDF5 file
file = h5py.File(
    "Data/market_data.h5",
    "r",
)

# Get the number of clients from sheet_name
num_clients = len(sheet_name)

# Set the number of iterations,rounds,epochs for federated learning
num_round = [20]
num_epochs = 10
num_iterations = 1

# # Initialize an empty similarity matrix to store similarity values for each pair of clients
# similarity_matrix_total1 = np.zeros((len(sheet_name), len(sheet_name)))
# similarity_matrix_total2 = np.zeros((len(sheet_name), len(sheet_name)))
# similarity_matrix_total3 = np.zeros((len(sheet_name), len(sheet_name)))

# Initialize an empty similarity matrix to store similarity values for each pair of clients for each iteration
similarity_matrix_total = np.zeros((len(sheet_name), len(sheet_name), num_iterations))


def train(model, device, train_loader, optimizer, epoch, criterion):
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

    # # Print and flush the output
    # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    logger = logging.getLogger(__name__)
    logger.info(f"(Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")


def test(model, device, client, test_inputs, test_targets):
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

    return r2


def data_processing(client, device):
    # Convert the client index to a string
    client_str = str(client)
    
    dataset = file[client_str][:]
    dataset = pd.DataFrame(dataset)

    # Read the column names from the attributes
    column_names = file[client_str].attrs["columns"]

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

    return train_inputs, train_loader, test_inputs, test_targets


def create_model(
    input_neurons, output_neurons, hidden_layers, neurons_per_layer, dropout
):
    # Define the neural network model
    model = Net(
        input_neurons, output_neurons, hidden_layers, neurons_per_layer, dropout
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    return model, criterion, optimizer




def main():
    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)

    device = torch.device("cpu")
    print("Using Device: ", device)

    # Initialize dictionary to store rankings for each region
    region_rankings = {region.strip(): [] for region in region_names}


    # Iterate through combinations and calculate rankings for each region
    for combination, avg_similarity in sorted_combo_avg_similarity:
        for region in combination:
            region_rankings[region.strip()].append((combination, avg_similarity))


    # Iterate through each region and perform federated learning
    for region, rankings in region_rankings.items():
        print(f"Federated learning for {region}:")
        for rank, (combination, _) in enumerate(rankings, start=1):
            print(f"Combination {rank}:", combination)
            # Perform federated learning for the current region and combination
            # Modify sheet_name to the current region (you may need to convert the region back to its corresponding index)
            sheet_name = [region_index for region_index, region_name in region_map.items() if region_name == region]

            # Call the federated learning function passing appropriate arguments

            # Initialize a dictionary to store the final round R2 values for each client
            final_round_r2 = {client: [] for client in sheet_name}


            for num_rounds in num_round:
                print(f"testing {num_rounds }")
                # Perform federated learning

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

                    # Set the parameters for the model
                    input_neurons = 25
                    output_neurons = 1
                    hidden_layers = 4
                    neurons_per_layer = 64
                    dropout = 0.3

                    # Initialize a shared global model
                    global_model = Net(
                        input_neurons, output_neurons, hidden_layers, neurons_per_layer, dropout
                    ).to(device)

                    # Initialize an empty list to store the client models for this round
                    client_models = []

                    for round in range(num_rounds):
                        # Initialize an empty list to store the client models for this round
                        client_models = []
                        print(f"Round {round + 1}/{num_rounds}")

                        for client in sheet_name:
                            (
                                train_inputs,
                                train_loader,
                                test_inputs,
                                test_targets,
                            ) = data_processing(client, device)

                            # Load the state dict of the global model to the client model
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

                            model.load_state_dict(global_model.state_dict())

                            for epoch in range(num_epochs):
                                train(model, device, train_loader, optimizer, epoch, criterion)

                            r2 = test(model, device, client, test_inputs, test_targets)

                            # Use model to generate predictions for the test dataset
                            client_models.append(model.state_dict())

                            # Save the R2 value for the current round and iteration
                            metrics[client]["r2"][iteration][round] = r2

                            # Record the final round R2 value for each client
                            if round == num_rounds - 1:
                                final_round_r2[client].append(r2)

                        # Average the weights across all clients after each round
                        averaged_weights = {
                            k: sum(d[k] for d in client_models) / num_clients
                            for k in client_models[0].keys()
                        }

                        # Update the global model
                        global_model.load_state_dict(averaged_weights)



            # Print the R2 values for each combination
            print(f"R2 values for combination {rank}:")
            for client, r2_values in final_round_r2.items():
                print(f"Client {client}: {r2_values}")

if __name__ == "__main__":
    main()
