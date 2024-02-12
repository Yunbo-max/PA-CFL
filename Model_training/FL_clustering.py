# -*- coding: utf-8 -*-
# @Author: Yunbo
# @Date:   2024-02-11 12:31:58
# @Last Modified by:   Yunbo
# @Last Modified time: 2024-02-12 00:00:41
import pickle
import xgboost as xgb
import numpy as np
from Data_prepare.Data_split import CustomDataset


# Read datasets from disk
with open('./Model_training/Data_prepare/datasets.pkl', 'rb') as f:
    loaded_datasets = pickle.load(f)

feature_importance_scores = []

for region, data in loaded_datasets.items():
    # Extract features and labels from the dataset
    train_loader = data['train']
    region_feature_importance_scores = []
    
    # Merge all data in the current region
    X_region, y_region = [], []
    for X_batch, y_batch in train_loader:
        X_region.extend(X_batch.numpy())
        if isinstance(y_batch, np.ndarray):
            y_region.extend(y_batch)
        else:
            y_region.append(y_batch.item())  # Convert scalar to item
        
    X_region = np.array(X_region)
    y_region = np.array(y_region)

    # Train XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

    # Convert list of lists to a 2D numpy array
    # Convert list of lists to a 2D numpy array
    X_region = np.array(X_region)

    # Debugging print statements
    print("Shape of X_region before reshaping:", X_region.shape)
    print("Content of X_region before reshaping:", X_region)

    # Reshape X_region to ensure it has two dimensions
    X_region = X_region.reshape(-1, len(X_region[0]))

    # Debugging print statements
    print("Shape of X_region after reshaping:", X_region.shape)
    print("Content of X_region after reshaping:", X_region)

    model.fit(X_region, y_region)
    # Store feature importance scores for this region
    feature_importance_scores.append(model.feature_importances_)

# Convert list of feature importance arrays to a numpy array
feature_importance_scores = np.array(feature_importance_scores)


