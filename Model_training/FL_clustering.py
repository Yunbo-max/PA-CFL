# -*- coding: utf-8 -*-
# @Author: Yunbo
# @Date:   2024-02-11 12:31:58
# @Last Modified by:   Yunbo
# @Last Modified time: 2024-02-11 16:42:29
import pickle
import xgboost as xgb
from Data_prepare.Data_split import CustomDataset

# Read datasets from disk
with open('./Model_training/Data_prepare/datasets.pkl', 'rb') as f:
    loaded_datasets = pickle.load(f)

feature_importance_scores = []

for region, data in loaded_datasets.items():
    # Extract features and labels from the dataset
    train_loader = data['train']
    
    # Iterate over the DataLoader to access the dataset
    for batch_idx, (X, y) in enumerate(train_loader):
        # Train XGBoost model
        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        model.fit(X, y)
        
        # Store feature importance scores
        feature_importance_scores.append(model.feature_importances_)

feature_importance_scores = np.array(feature_importance_scores)
print(feature_importance_scores)



