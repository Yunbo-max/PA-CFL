# -*- coding: utf-8 -*-
# @Author: Yunbo
# @Date:   2024-02-11 15:39:33
# @Last Modified by:   Yunbo
# @Last Modified time: 2024-02-12 00:34:54
import pickle
import xgboost as xgb
import numpy as np
from Data_prepare.Data_split import CustomDataset

# Read datasets from disk
with open('./Model_training/Data_prepare/datasets.pkl', 'rb') as f:
    loaded_datasets = pickle.load(f)

feature_importance_scores = []

for region, data in loaded_datasets.items():
    # Extract features and labels from the training dataset
    train_data = data['train']
    features = train_data.iloc[:, :-1]  # All columns except the last one
    labels = train_data.iloc[:, -1]      # Last column
    
    # Train XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    model.fit(features, labels)

        # Store feature importance scores for this batch
    feature_importance_scores.append(model.feature_importances_)

feature_importance_scores = np.array(feature_importance_scores)
print(feature_importance_scores)


