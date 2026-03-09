# -*- coding: utf-8 -*-
# PyTorch dataset classes shared across all methods.

from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """Dataset for raw numpy features and labels."""
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class RegressionDataset(Dataset):
    """Dataset for tensor features and labels."""
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
