# -*- coding: utf-8 -*-
# Step 1: Feature Importance Calculation using XGBoost
# Computes per-client feature importance scores for clustering.

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split


def compute_feature_importance(features, labels, test_size=0.2, random_state=42):
    """Compute XGBoost feature importance for a single client's data.

    Args:
        features: np.ndarray of shape (n_samples, n_features)
        labels: np.ndarray of shape (n_samples,)

    Returns:
        importance_scores: np.ndarray of shape (n_features,)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state)

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=random_state
    )
    model.fit(X_train, y_train)

    importance_scores = model.feature_importances_
    return importance_scores


def compute_all_feature_importances(datasets, region_ids):
    """Compute feature importance for all active clients.

    Args:
        datasets: dict of region_name -> {train_features, train_labels, ...}
        region_ids: list of (region_name, region_id) tuples

    Returns:
        importance_matrix: np.ndarray of shape (n_clients, n_features)
        client_names: list of region names
    """
    importance_matrix = []
    client_names = []

    for region_name, _ in region_ids:
        if region_name not in datasets:
            continue
        data = datasets[region_name]
        features = data['train_features'].numpy()
        labels = data['train_labels'].numpy()

        scores = compute_feature_importance(features, labels)
        importance_matrix.append(scores)
        client_names.append(region_name)

    return np.array(importance_matrix), client_names
