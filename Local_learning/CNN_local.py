# -*- coding: utf-8 -*-
# @Author: Yunbo
# @Date:   2024-03-10 09:30:26
# @Last Modified by:   Yunbo
# @Last Modified time: 2024-10-05 22:31:28


# -*- coding: utf-8 -*-
# @Author: Yunbo
# @Date:   2024-01-24 10:28:47
# @Last Modified by:   Yunbo
# @Last Modified time: 2024-03-10 09:29:56
# -*- coding = utf-8 -*-
# @time:06/07/2023 12:02
# Author:Yunbo Long
# @File:TF.py
# @Software:PyCharm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import warnings
from sklearn import metrics
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import warnings
from sklearn import metrics
import wandb
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
import wandb
from sklearn.metrics import r2_score

region_map = {
    0: 'Southeast Asia',
    1: 'South Asia',
    2: 'Oceania',
    3: 'Eastern Asia',
    5: 'West of USA',
    6: 'US Center',
    7: 'West Africa',
    9: 'North Africa',
    10: 'Western Europe',
    12: 'Central America',
    14: 'South America',
    16: 'Southern Europe',
    17: 'East of USA',
    22: 'South of USA'
}

# Define the CNN model for regression
class SalesPredictionCNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SalesPredictionCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension for CNN
        
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        x = torch.mean(x, dim=2)  # Global Average Pooling
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x




warnings.filterwarnings('ignore')

import h5py

# Open the HDF5 file
file = h5py.File('/Users/yunbo/Documents/GitHub/PFL_Optimiozation/Clustering_analysis copy/market_data.h5', 'r')

# Read the dataset using the current sheet name
sheet_name = ['0','1', '2', '3', '5', '6', '7', '9', '10', '12', '14', '16', '17', '22']
# Initialize Weights and Biases


for sheet_names in sheet_name:
    value = region_map[float(sheet_names)]
    config = {"region": value}
    with wandb.init(project='CFL_transformer_local', config=config) as run:

        # Read the dataset using the current sheet name
        dataset = file[sheet_names][:]
        dataset = pd.DataFrame(dataset)

        # Read the column names from the attributes
        column_names = file[sheet_names].attrs['columns']

        # Assign column names to the dataset
        dataset.columns = column_names

        print(dataset.columns)

        dataset = dataset.drop(columns=['Region Index'])

        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        print(dataset.shape)

        

        # Preprocess the data
        # # Preprocess the data
        # train_data = dataset # Drop the last 30 rows
        # xs = train_data.drop(['Sales'], axis=1)
        # ys = train_data['Sales']  # Use the updated train_data for ys
        # xs_train, xs_test, ys_train, ys_test = train_test_split(xs, ys, test_size=0.2, random_state=42)



        # Ensure that the dataset is ordered by 'order_year' and 'order_month'
        dataset = dataset.sort_values(by=['order_year', 'order_month'])

        # Preprocess the data
        train_data = dataset

        # Split based on time sequence
        split_index = int(len(train_data) * 0.8)  # 80% for training, 20% for testing
        train_data = dataset.iloc[:split_index]
        test_data = dataset.iloc[split_index:]

        # Separate features and target variable
        xs_train = train_data.drop(['Sales'], axis=1)
        ys_train = train_data['Sales']
        xs_test = test_data.drop(['Sales'], axis=1)
        ys_test = test_data['Sales']


        # Scale the input features
        scaler = MinMaxScaler()
        xs_train = scaler.fit_transform(xs_train)
        xs_test = scaler.transform(xs_test)

        # Convert the data to tensors
        train_inputs = torch.tensor(xs_train, dtype=torch.float32)
        train_targets = torch.tensor(ys_train.values, dtype=torch.float32)
        test_inputs = torch.tensor(xs_test, dtype=torch.float32)
        test_targets = torch.tensor(ys_test.values, dtype=torch.float32)

        # Adjust the batch size
        batch_size = 64

        # Create DataLoader for batch processing
        train_dataset = TensorDataset(train_inputs, train_targets)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        test_dataset = TensorDataset(test_inputs, test_targets)
        testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        print(train_inputs.shape)
        print(train_targets.shape)
        print(test_inputs.shape)
        print(test_targets.shape)


        # Define the hyperparameters
        # Define the hyperparameters
        # Initialize the CNN model, loss function, and optimizer
        input_dim = train_inputs.size(1)  # Make sure this matches your input features
        learning_rate = 0.001
        output_dim = 1  # Assuming you are predicting sales (regression)
        model = SalesPredictionCNN(input_dim, output_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Rest of the training loop remains the same as before


        # # Save the initial parameters
        # torch.save(model.state_dict(), 'initial_parameters.pth')


        num_epochs = 100

        def evaluate_model1(net, testloader):
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

                    if batch_size < testloader.batch_size:
                        continue  # Skip the last incomplete batch

                    outputs = net(inputs)

                    batch_loss = criterion(outputs.squeeze(), batch_targets.float())
                    loss += batch_loss.item() * batch_size

                    if isinstance(outputs, torch.Tensor):  # Check if outputs is a tensor
                        outputs = outputs.squeeze().cpu().tolist()  # Convert to list if tensor
                    else:
                        outputs = [outputs]  # Wrap single float in a list

                    # print("Outputs:", outputs)
                    # print("Batch Targets:", batch_targets.cpu().tolist())

                    predictions.extend(outputs)  # Extend predictions with the batch predictions
                    targets.extend(batch_targets.cpu().tolist())  # Extend targets with the batch targets

            predictions = np.array(predictions)
            targets = np.array(targets)
            loss /= num_samples

            return predictions, targets, loss


        # Train the transformer model
        # Train the transformer model
        losses = []
        num_batches = len(train_loader)
        for epoch in range(num_epochs):
            batch_losses = []
            for batch_idx, (batch_inputs, batch_targets) in enumerate(train_loader):
                # Skip the last incomplete batch
                if batch_idx == num_batches - 1 and len(batch_inputs) < batch_size:
                    continue

                optimizer.zero_grad()
                outputs = model(batch_inputs)
                loss = criterion(outputs.squeeze(), batch_targets)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())

            epoch_loss = np.mean(batch_losses)
            losses.append(epoch_loss)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
            wandb.log({'Training Loss': epoch_loss})  # Log the training loss to Wandb

        # Evaluate the model
        # Evaluation function


            # Evaluate the model on the test set
            predictions, targets, loss = evaluate_model1(model, testloader)


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
                'Test Loss': loss
            })

            performance_metrics = {
                "rmse": float(rmse_val),
                "r_squared": float(r2_val),
                "mae": float(mae_val),
                "mse": float(mse_val),
                "loss": float(loss)
            }

