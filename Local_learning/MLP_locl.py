# -*- coding: utf-8 -*-
# @Author: Yunbo
# @Date:   2024-03-10 08:14:24
# @Last Modified by:   Yunbo
# @Last Modified time: 2024-10-02 00:09:31

# -*- coding: utf-8 -*-
# @Author: Yunbo
# @Date:   2024-01-24 10:28:47
# @Last Modified by:   Yunbo
# @Last Modified time: 2024-02-13 14:11:57
# -*- coding = utf-8 -*-
# @time:07/07/2023 22:55
# Author:Yunbo Long
# @File:test.py
# @Software:PyCharm
# -*- coding = utf-8 -*-
# @time:05/07/2023 13:07
# Author:Yunbo Long
# @File:CNN_centralised_5_quantity.py
# @Software:PyCharm
# -*- coding = utf-8 -*-
# @time:05/05/2023 11:20
# Author:Yunbo Long
# @File:CNN.py
# @Software:PyCharm
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split,cross_val_score, cross_val_predict
from sklearn import svm,metrics,tree,preprocessing,linear_model
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.linear_model import Ridge,LinearRegression,LogisticRegression,ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier, GradientBoostingRegressor,BaggingClassifier,ExtraTreesClassifier
from sklearn.metrics import accuracy_score,mean_squared_error,recall_score,confusion_matrix,f1_score,roc_curve, auc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns



# Hiding the warnings
warnings.filterwarnings('ignore')


import wandb

import h5py

# Open the HDF5 file
# Open the HDF5 file
file = h5py.File('/Users/yunbo/Documents/GitHub/Demand-Forecasting-with-PFL-methods/Explotary data analysis/market_data_test.h5', 'r')

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

sheet_name = [str(i) for i in range(23)]

for sheet_names in sheet_name:
    value = region_map[float(sheet_names)]
    config = {"region": value}
    with wandb.init(project='CFL_local_MLP3', config=config) as run:

        # Read the dataset using the current sheet name
        dataset = file[sheet_names][:]
        dataset = pd.DataFrame(dataset)

        # Read the column names from the attributes
        column_names = file[sheet_names].attrs['columns']

        # Assign column names to the dataset
        dataset.columns = column_names

        dataset = dataset.drop(columns=['Region Index'])

        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        # print(dataset.head(30))

        # Preprocess the data
        train_data = dataset # Drop the last 30 rows
        xs = train_data.drop(['Sales'], axis=1)
        ys = train_data['Sales']  # Use the updated train_data for ys
        xs_train, xs_test, ys_train, ys_test = train_test_split(xs, ys, test_size=0.3, random_state=42)

        # xq=train_data.loc[:, train_data.columns != 'Order Item Quantity']
        # yq=train_data['Order Item Quantity']
        # xq_train, xq_test,yq_train,yq_test = train_test_split(xq,yq,test_size = 0.3, random_state = 42)

        scaler=MinMaxScaler()
        xs_train=scaler.fit_transform(xs_train)
        xs_test=scaler.transform(xs_test)
        # xq_train=scaler.fit_transform(xq_train)
        # xq_test=scaler.transform(xq_test)


        train_inputs = torch.tensor(xs_train, dtype=torch.float32)
        train_targets = torch.tensor(ys_train.values, dtype=torch.float32)
        test_inputs = torch.tensor(xs_test, dtype=torch.float32)
        test_targets = torch.tensor(ys_test.values, dtype=torch.float32)

        print(train_inputs.shape)
        print(train_targets.shape)

        import torch
        import torch.nn as nn
        import torch.optim as optim
        import numpy as np
        import pandas as pd
        import numpy as np
        import xgboost as xgb
        import warnings
        from sklearn.model_selection import train_test_split,cross_val_score, cross_val_predict
        from sklearn import svm,metrics,tree,preprocessing,linear_model
        from sklearn.metrics import mean_absolute_error, mean_squared_error


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
        learning_rate = 0.001
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # # Save the initial parameters
        # torch.save(model.state_dict(), 'initial_parameters.pth')


        num_epochs = 100
        batch_size = 32
        evaluation_interval = 1  # Set the evaluation interval (e.g., every 5 epochs)

        import seaborn as sns
        import matplotlib.pyplot as plt
        import wandb

        losses = []
        train_rmse_list = []
        test_rmse_list = []
        mae_train_list = []
        rmse_train_list = []
        mape_train_list = []
        mse_train_list = []
        r2_train_list = []
        mae_test_list = []
        rmse_test_list = []
        mape_test_list = []
        mse_test_list = []
        r2_test_list = []

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

            model.eval()
            train_outputs = model(train_inputs)
            train_rmse = torch.sqrt(torch.mean((train_outputs.squeeze() - train_targets) ** 2)).item()

            pred_s_train = train_outputs.squeeze().detach().numpy()
            mae_train = mean_absolute_error(train_targets.detach().numpy(), pred_s_train)
            rmse_train = np.sqrt(mean_squared_error(train_targets.detach().numpy(), pred_s_train))
            mape_train = np.mean(
                np.abs((train_targets.detach().numpy() - pred_s_train) / train_targets.detach().numpy())) * 100
            mse_train = mean_squared_error(train_targets.detach().numpy(), pred_s_train)
            r2_train = metrics.r2_score(train_targets.detach().numpy(), pred_s_train)

            mae_test, rmse_test, mape_test, mse_test, r2_test = None, None, None, None, None

            model.eval()
            if epoch % evaluation_interval == 0:
                test_outputs = model(test_inputs)
                test_loss = criterion(test_outputs, test_targets.unsqueeze(1))
                test_rmse = torch.sqrt(torch.mean((test_outputs.squeeze() - test_targets) ** 2)).item()

                pred_s_test = test_outputs.squeeze().detach().numpy()
                mae_test = mean_absolute_error(test_targets.detach().numpy(), pred_s_test)
                rmse_test = np.sqrt(mean_squared_error(test_targets.detach().numpy(), pred_s_test))
                mape_test = np.mean(
                    np.abs((test_targets.detach().numpy() - pred_s_test) / test_targets.detach().numpy())) * 100
                mse_test = mean_squared_error(test_targets.detach().numpy(), pred_s_test)
                r2_test = metrics.r2_score(test_targets.detach().numpy(), pred_s_test)

                mae_test_list.append(mae_test)
                rmse_test_list.append(rmse_test)
                mape_test_list.append(mape_test)
                mse_test_list.append(mse_test)
                r2_test_list.append(r2_test)

            wandb.log({
                'Train Loss': epoch_loss,
                'Train RMSE': train_rmse,
                'MAE (train data)': mae_train,
                'RMSE (train data)': rmse_train,
                'MAPE (train data)': mape_train,
                'MSE (train data)': mse_train,
                'R2 (train data)': r2_train,
                'Test Loss': test_loss.item(),
                'MAE (test data)': mae_test,
                'RMSE (test data)': rmse_test,
                'MAPE (test data)': mape_test,
                'MSE (test data)': mse_test,
                'R2 (test data)': r2_test
            })

            region_value = region_map[float(sheet_names)]

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
            plt.savefig('scatter_plot.png')
            wandb.log({'Scatter Plot': wandb.Image('scatter_plot.png')})
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
            plt.savefig('distribution_plot.png')
            wandb.log({'Distribution Plot': wandb.Image('distribution_plot.png')})
            plt.close()

    print('Train Loss:', epoch_loss)
    print('Train RMSE:', train_rmse)
    print('MAE Value (train data):', mae_train)
    print('RMSE (train data):', rmse_train)
    print('MAPE (train data):', mape_train)
    print('MSE (train data):', mse_train)
    print('R2 (train data):', r2_train)
    print('Test Loss:', test_loss.item())
    print('MAE Value (test data):', mae_test)
    print('RMSE (test data):', rmse_test)
    print('MAPE (test data):', mape_test)
    print('MSE (test data):', mse_test)
    print('R2 (test data):', r2_test)
    print('\n')

