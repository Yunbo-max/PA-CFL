# -*- coding: utf-8 -*-
# @Author: Yunbo
# @Date:   2024-02-11 12:31:58
# @Last Modified by:   Yunbo
# @Last Modified time: 2024-02-12 00:51:54
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
    
    for batch_idx, (X, y) in enumerate(train_loader):
        # Reshape X to ensure it has two dimensions
        X = X.reshape(-1, 42)
        
        # Print the shapes of X and y
        print(f"Shapes of X and y in batch {batch_idx}: {X.shape}, {y.shape}")

        # Train XGBoost model
        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        model.fit(X, y)
        
        # Store feature importance scores for this batch
        feature_importance_scores.append(model.feature_importances_)

# Convert list of feature importance arrays to a numpy array
feature_importance_scores = np.array(feature_importance_scores)

# Calculate the average feature importance across all batches
average_feature_importance = np.mean(feature_importance_scores, axis=0)

print("Average Feature Importance Scores:")
print(average_feature_importance)




