# -*- coding: utf-8 -*-
# @Author: Yunbo
# @Date:   2024-02-11 12:23:32
# @Last Modified by:   Yunbo
# @Last Modified time: 2024-02-11 16:44:24

import pickle
from torch.utils.data import Dataset, DataLoader
from Data_split import CustomDataset


# Read datasets from disk
with open('datasets.pkl', 'rb') as f:
    loaded_datasets = pickle.load(f)

# Accessing the datasets for each Order Region
for region, data in loaded_datasets.items():
    print(f"Region: {region}")
    print(f"Training samples: {len(data['train'])}")
    print(f"Testing samples: {len(data['test'])}")
    print()

# Example usage of accessing a specific dataset
train_loader = DataLoader(loaded_datasets['Northern Europe']['train'], batch_size=64, shuffle=True)
test_loader = DataLoader(loaded_datasets['Northern Europe']['test'], batch_size=64, shuffle=False)
