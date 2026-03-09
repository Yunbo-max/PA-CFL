# -*- coding: utf-8 -*-
# Shared evaluation metrics for all methods.

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(targets, predictions):
    """Compute all regression metrics. Returns dict."""
    targets = np.array(targets).flatten()
    predictions = np.array(predictions).flatten()
    return {
        "rmse": float(np.sqrt(mean_squared_error(targets, predictions))),
        "mae": float(mean_absolute_error(targets, predictions)),
        "mape": float(np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100),
        "mse": float(mean_squared_error(targets, predictions)),
        "r_squared": float(r2_score(targets, predictions)),
    }


def evaluate_model(model, dataloader):
    """Evaluate a model on a dataloader. Returns (predictions, targets, avg_loss)."""
    criterion = torch.nn.MSELoss()
    model.eval()
    predictions, targets = [], []
    total_loss, num_samples = 0.0, 0

    with torch.no_grad():
        for inputs, batch_targets in dataloader:
            batch_size = inputs.size(0)
            num_samples += batch_size
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), batch_targets.float())
            total_loss += loss.item() * batch_size
            predictions.append(outputs.squeeze().cpu().numpy())
            targets.append(batch_targets.cpu().numpy())

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
    return predictions, targets, avg_loss
