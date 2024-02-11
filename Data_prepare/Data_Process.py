# -*- coding: utf-8 -*-
# @Author: Yunbo
# @Date:   2024-02-11 00:22:54
# @Last Modified by:   Yunbo
# @Last Modified time: 2024-02-11 14:39:23

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os

# Read the dataset
dataset = pd.read_csv('./Data_collection/DataCoSupplyChainDataset.csv', encoding='latin-1')

# Display options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Display the first 30 rows of the dataset
print(dataset.head(30))

# # Define a PyTorch dataset class
# class CustomDataset(Dataset):
#     def __init__(self, data):
#         self.data = data
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         # Implement your own data retrieval logic here
#         # For example, you can return a tuple of features and labels
#         return self.data.iloc[idx, :-1], self.data.iloc[idx, -1]

# # Split the dataset according to 'Order Region' and save into PyTorch datasets
# datasets = {}
# for region, data_region in dataset.groupby('Order Region'):
#     # Split data into features and labels
#     features = data_region.drop(columns=['Order Region'])
#     labels = data_region['Order Region']
    
#     # Split into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
#     # Create PyTorch datasets
#     train_dataset = CustomDataset(pd.concat([X_train, y_train], axis=1))
#     test_dataset = CustomDataset(pd.concat([X_test, y_test], axis=1))
    
#     # Store datasets
#     datasets[region] = {'train': train_dataset, 'test': test_dataset}


# # Specify the directory where you want to save the file
# folder_path = 'Data_prepare'

# # Ensure that the directory exists, create it if necessary
# os.makedirs(folder_path, exist_ok=True)

# # Save datasets to disk in the specified folder
# file_path = os.path.join(folder_path, 'datasets.pkl')
# with open(file_path, 'wb') as f:
#     pickle.dump(datasets, f)



