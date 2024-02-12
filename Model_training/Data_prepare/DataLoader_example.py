# -*- coding: utf-8 -*-
# @Author: Yunbo
# @Date:   2024-02-11 12:23:32
# @Last Modified by:   Yunbo
# @Last Modified time: 2024-02-12 11:07:46

import pickle
from Data_split import CustomDataset
from torch.utils.data import DataLoader

# Load datasets from the pickle file
file_path = 'Model_training/Data_prepare/datasets.pkl'
with open(file_path, 'rb') as f:
    datasets = pickle.load(f)


train_dataset_canada = datasets['Canada']['train']
train_features, _ = train_dataset_canada[:]

# Print the shape of the matrix
print("Shape of the feature matrix:", train_features.shape)
# # Iterate over each region
# for region, data in datasets.items():
#     print(f"Region: {region}")
    
#     # Print train data
#     print("Train data:")
#     train_dataset = data['train']
#     for features, labels in train_dataset:
#         print(f"Features: {features.shape}, Labels: {labels.shape}")


# # # Example usage of accessing a specific dataset
# # train_loader = DataLoader(loaded_datasets['Northern Europe']['train'], batch_size=64, shuffle=True)
# # test_loader = DataLoader(loaded_datasets['Northern Europe']['test'], batch_size=64, shuffle=False)
