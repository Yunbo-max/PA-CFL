# -*- coding: utf-8 -*-
# Step 2: Differential Privacy
# Adds calibrated Laplace noise to feature importance scores.

import numpy as np


def compute_local_sensitivity(features, labels, compute_importance_fn,
                              n_samples=10, random_state=42):
    """Estimate local sensitivity by leave-one-out on a subset.

    Args:
        features: np.ndarray (n_samples, n_features)
        labels: np.ndarray (n_samples,)
        compute_importance_fn: callable that returns importance scores
        n_samples: number of samples to check for sensitivity estimation

    Returns:
        sensitivity: float (max change in importance from removing one sample)
    """
    rng = np.random.RandomState(random_state)
    full_importance = compute_importance_fn(features, labels)

    max_diff = 0.0
    indices = rng.choice(len(features), size=min(n_samples, len(features)), replace=False)

    for idx in indices:
        mask = np.ones(len(features), dtype=bool)
        mask[idx] = False
        loo_importance = compute_importance_fn(features[mask], labels[mask])
        diff = np.max(np.abs(full_importance - loo_importance))
        max_diff = max(max_diff, diff)

    return max_diff


def add_laplace_noise(importance_scores, epsilon, sensitivity):
    """Add Laplace noise to feature importance scores for differential privacy.

    Args:
        importance_scores: np.ndarray of shape (n_features,) or (n_clients, n_features)
        epsilon: float, privacy budget (higher = less noise)
        sensitivity: float, local sensitivity

    Returns:
        noisy_scores: np.ndarray, same shape as importance_scores
    """
    sigma = sensitivity / epsilon
    noise = np.random.laplace(loc=0, scale=sigma, size=importance_scores.shape)
    return importance_scores + noise


def apply_differential_privacy(importance_matrix, epsilon=10.0, sensitivity=None):
    """Apply differential privacy to the full importance matrix.

    Args:
        importance_matrix: np.ndarray (n_clients, n_features)
        epsilon: privacy budget
        sensitivity: if None, estimated from data

    Returns:
        noisy_matrix: np.ndarray (n_clients, n_features)
    """
    if sensitivity is None:
        # Use max absolute value as a conservative sensitivity estimate
        sensitivity = np.max(np.abs(importance_matrix)) / len(importance_matrix)

    return add_laplace_noise(importance_matrix, epsilon, sensitivity)
