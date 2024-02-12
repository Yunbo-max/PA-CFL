# -*- coding: utf-8 -*-
# @Author: Yunbo
# @Date:   2024-02-12 13:58:32
# @Last Modified by:   Yunbo
# @Last Modified time: 2024-02-12 13:58:38
# -*- coding = utf-8 -*-
# @time:05/07/2023 18:11
# Author:Yunbo Long
# @File:client_CNN6.py
# @Software:PyCharm
# Import the necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import flwr as fl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# Load and preprocess your dataset
# Modify this part according to your dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from tensorboardX import SummaryWriter
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns

import lightgbm as lgb
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn import svm, metrics, tree, preprocessing, linear_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression, ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, \
    BaggingClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, recall_score, confusion_matrix, f1_score, roc_curve, auc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from tensorboardX import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import wandb
import matplotlib.pyplot as plt

# Hiding the warnings
warnings.filterwarnings('ignore')
# Hiding the warnings
warnings.filterwarnings('ignore')


# # Initialize Weights and Biases
# wandb.init(project="CNN", name=f"Sheet_{random_sheet_name}")


import wandb

import h5py

# Open the HDF5 file
file = h5py.File('E:\Federated_learning_flower\experiments\Presentation\market_data.h5', 'r')

region_map = {
    0: 'Southeast Asia',
    1: 'South Asia',
    2: 'Oceania',
    3: 'Eastern Asia',
    4: 'West Asia',
    5: 'West of USA',
    6: 'US Center',
    7: 'West Africa',
    8: 'Central Africa',
    9: 'North Africa',
    10: 'Western Europe',
    11: 'Northern Europe',
    12: 'Central America',
    13: 'Caribbean',
    14: 'South America',
    15: 'East Africa',
    16: 'Southern Europe',
    17: 'East of USA',
    18: 'Canada',
    19: 'Southern Africa',
    20: 'Central Asia',
    21: 'Eastern Europe',
    22: 'South of USA'
}

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

# Print the value for a given key
key = 3
value = region_map[key]
print(value)

sheet_name = '19'

value = region_map[float(sheet_name)]
config = {"region": value}
wandb.init(project='SCAIL4', config=config)

# Read the dataset using the current sheet name
dataset = file[sheet_name][:]
dataset = pd.DataFrame(dataset)

# Read the column names from the attributes
column_names = file[sheet_name].attrs['columns']

# Assign column names to the dataset
dataset.columns = column_names

dataset = dataset.drop(columns=['Region Index'])

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
print(dataset.head(30))

# Preprocess the data
train_data = dataset
xs = train_data.drop(['Sales'], axis=1)
ys = dataset['Sales']
xs_train, xs_test, ys_train, ys_test = train_test_split(xs, ys, test_size=0.3, random_state=42)

scaler = MinMaxScaler()
xs_train = scaler.fit_transform(xs_train)
xs_test = scaler.transform(xs_test)

train_inputs = torch.tensor(xs_train, dtype=torch.float32)
train_targets = torch.tensor(ys_train.values, dtype=torch.float32)
test_inputs = torch.tensor(xs_test, dtype=torch.float32)
test_targets = torch.tensor(ys_test.values, dtype=torch.float32)

print(train_inputs.shape)
print(train_targets.shape)


# Define the dataset class
class RegressionDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


# Create the dataset objects
train_dataset = RegressionDataset(train_inputs, train_targets)
test_dataset = RegressionDataset(test_inputs, test_targets)

# Create the data loaders
trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Create the model
# Create the model
class Net(nn.Module):
    def __init__(self, input_neurons, output_neurons, hidden_layers, neurons_per_layer, dropout):
        super().__init__()

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


input_neurons = train_inputs.shape[1]
output_neurons = 1
hidden_layers = 4
neurons_per_layer = 64
dropout = 0.3
model = Net(input_neurons, output_neurons, hidden_layers, neurons_per_layer, dropout)

criterion = nn.MSELoss()
learning_rate = 0.005
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Define performance metrics functions
def rmse(targets, predictions):
    return float(np.sqrt(mean_squared_error(targets, predictions)))


def r_squared(targets, predictions):
    return float(r2_score(targets, predictions))


def mae(targets, predictions):
    return float(mean_absolute_error(targets, predictions))


def mse(targets, predictions):
    return float(mean_squared_error(targets, predictions))


# Define the client class
class RegressionClient(fl.client.NumPyClient):
    def __init__(self):
        self.total_communication_time = 0.0

    def get_parameters(self, config=None):
        return [param.detach().numpy().astype('float32') for param in model.parameters()]

    def fit(self, parameters, config):
        for param, new_param in zip(model.parameters(), parameters):
            param.data.copy_(torch.from_numpy(new_param).float())  # Update local model parameters

        num_epochs = 5
        batch_size = 32
        evaluation_interval = 1  # Set the evaluation interval (e.g., every 5 epochs)

        import seaborn as sns
        import matplotlib.pyplot as plt
        import wandb
        losses = []

        start_time = time.time()  # Start time for the current epoch
        for epoch in range(num_epochs):
            permutation = torch.randperm(train_inputs.size()[0])

            batch_losses = []

            for i in range(0, train_inputs.size()[0], batch_size):
                indices = permutation[i:i + batch_size]
                batch_inputs, batch_targets = train_inputs[indices], train_targets[indices]

                optimizer.zero_grad()
                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_targets.unsqueeze(1))
                loss.backward()
                optimizer.step()

                batch_losses.append(loss.item())

            epoch_loss = np.mean(batch_losses)
            losses.append(epoch_loss)

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        elapsed_time = time.time() - start_time  # Elapsed time for the current epoch

        wandb.log({"Communication Time": elapsed_time})

        return [param.detach().numpy().astype('float32') for param in model.parameters()], len(train_dataset), {}

    def evaluate(self, parameters, config=None):
        """Evaluate parameters on the locally held test set."""
        # Update local model parameters
        # Update local model parameters
        for param, new_param in zip(model.parameters(), parameters):
            param.data.copy_(torch.from_numpy(new_param).float())

        testloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

        # Evaluation function
        def evaluate_model(net, testloader):
            criterion = torch.nn.MSELoss()  # Use Mean Squared Error (MSE) for regression
            net.eval()
            predictions = []
            targets = []
            loss = 0.0
            num_samples = 0

            with torch.no_grad():
                for inputs, batch_targets in testloader:
                    batch_size = inputs.size(0)
                    num_samples += batch_size
                    outputs = net(inputs)
                    batch_loss = criterion(outputs.squeeze(), batch_targets.float())
                    loss += batch_loss.item() * batch_size
                    predictions.append(outputs.numpy())
                    targets.append(batch_targets.numpy())

            predictions = np.concatenate(predictions)
            targets = np.concatenate(targets)
            loss /= num_samples

            return predictions, targets, loss

        # Evaluate the model on the test set
        predictions, targets, loss = evaluate_model(model, testloader)

        pred_s_test = predictions.squeeze()



        # Calculate evaluation metrics
        rmse_val = np.sqrt(mean_squared_error(targets, predictions))
        mae_val = mean_absolute_error(targets, predictions)
        mape_val = np.mean(np.abs((targets - predictions) / targets)) * 100
        mse_val = mean_squared_error(targets, predictions)
        r2_val = r2_score(targets, predictions)

        # Print metrics
        print("Performance Metrics:")
        print(f"RMSE (test data): {rmse_val}")
        print(f"MAE (test data): {mae_val}")
        print(f"MAPE (test data): {mape_val}")
        print(f"MSE (test data): {mse_val}")
        print(f"R2 (test data): {r2_val}")
        print(f"loss (test data): {loss}")

        wandb.log({
            'RMSE (test data)': rmse_val,
            'MAE (test data)': mae_val,
            'MAPE (test data)': mape_val,
            'MSE (test data)': mse_val,
            'R2 (test data)': r2_val,
            'loss(test data)': loss
        })

        region_value = region_map[float(sheet_name)]

        # Scatter Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(test_targets.detach().numpy(), pred_s_test, label=region_value)
        plt.xlabel('Actual Sales')
        plt.ylabel('Predicted Sales')
        plt.title('Scatter Plot: Actual vs Predicted Sales')

        # Add line of best fit
        coefficients = np.polyfit(test_targets.detach().numpy(), pred_s_test, 1)
        poly_line = np.polyval(coefficients, test_targets.detach().numpy())
        plt.plot(test_targets.detach().numpy(), poly_line, color='red', label='Line of Best Fit')

        plt.legend()
        plt.tight_layout()
        plt.savefig('scatter_plot19.png')
        wandb.log({'Scatter Plot': wandb.Image('scatter_plot19.png')})
        plt.close()

        # Distribution Plot
        plt.figure(figsize=(8, 6))
        residuals = test_targets.detach().numpy() - pred_s_test
        sns.histplot(residuals, kde=True, label=region_value)
        plt.xlabel('Residuals')
        plt.ylabel('Density')
        plt.title('Distribution of Residuals')
        plt.legend()
        plt.tight_layout()
        plt.savefig('distribution_plot19.png')
        wandb.log({'Distribution Plot': wandb.Image('distribution_plot19.png')})
        plt.close()

        performance_metrics = {
            "rmse": float(rmse_val),
            "r_squared": float(r2_val),
            "mae": float(mae_val),
            "mse": float(mse_val),
            "loss": float(loss)
        }
        return loss, len(test_dataset), performance_metrics


# Create the client and start the client
fl.client.start_numpy_client(
    server_address="127.0.0.1:8086",
    client=RegressionClient()
)
