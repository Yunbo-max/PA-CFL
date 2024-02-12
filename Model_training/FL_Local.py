# -*- coding: utf-8 -*-
# @Author: Yunbo
# @Date:   2024-02-11 15:39:33
# @Last Modified by:   Yunbo
# @Last Modified time: 2024-02-12 12:06:47
import pickle
import xgboost as xgb
import numpy as np
from Data_prepare.Data_split import CustomDataset
import pandas as pd
import networkx as nx
from sklearn.metrics import jaccard_score
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from grakel.kernels import GraphletSampling
from grakel.utils import graph_from_networkx
from sklearn.cluster import AgglomerativeClustering
import h5py
import xgboost as xgb
from scipy.stats import wasserstein_distance
from scipy.stats import wasserstein_distance


# Load datasets from the pickle file
file_path = 'Model_training/Data_prepare/datasets.pkl'
with open(file_path, 'rb') as f:
    datasets = pickle.load(f)

# Define a list to store feature importance scores
feature_importance_scores = []

region_map = {
    0: "Southeast Asia",
    1: "South Asia",
    2: "Oceania",
    3: "Eastern Asia",
    4: "West Asia",
    5: "West of USA",
    6: "US Center",
    7: "West Africa",
    8: "Central Africa",
    9: "North Africa",
    10: "Western Europe",
    11: "Northern Europe",
    12: "Central America",
    13: "Caribbean",
    14: "South America",
    15: "East Africa",
    16: "Southern Europe",
    17: "East of USA",
    18: "Canada",
    19: "Southern Africa",
    20: "Central Asia",
    21: "Eastern Europe",
    22: "South of  USA",
}

sheet_names = ['0', '1', '2', '3', '5', '6', '7', '9', '10', '12', '14', '16', '17', '22']


region_names = [region_map[int(sheet)].strip() for sheet in sheet_names]


# Iterate over each region
for region, data in datasets.items():
    print(region)
    if region.strip() in region_names:


        print(f"Training XGBoost model for region: {region}")
        
        # Access dataset for a specific region
        train_dataset = data['train']
        
        # Get train data for the region
        train_features, train_labels = train_dataset[:]
        print(train_features.shape)
        
        # Train XGBoost model
        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        model.fit(train_features, train_labels)
        
        # Store feature importance scores for this region
        feature_importance_scores.append(model.feature_importances_)
        
        # Print the shape of feature importance scores
        print(f"Shape of feature importance scores for region {region}: {model.feature_importances_.shape}")


# Convert list of feature importance arrays to a numpy array
feature_importance_scores = np.array(feature_importance_scores)


num_regions = len(feature_importance_scores)


similarity_matrix_xgboost = np.zeros((num_regions, num_regions))

# Reverse the label encoding to get the original region names
reverse_mapping = {v: k for k, v in region_map.items()}
# List to store region names
region_names = region_names
print(len(region_names))


# Function to normalize feature importance scores
def normalize_feature_importance(feature_importance_scores):
    max_scores = np.max(feature_importance_scores, axis=1)
    normalized_scores = feature_importance_scores / max_scores[:, np.newaxis]
    return normalized_scores

# Normalize feature importance scores
normalized_feature_importance = normalize_feature_importance(feature_importance_scores)


def calculate_emd(prob_dist1, prob_dist2):
    return wasserstein_distance(prob_dist1, prob_dist2)


# Calculate similarity matrix using Earth Mover's Distance
for i in range(num_regions):
    for j in range(i + 1, num_regions):
        emd_value = calculate_emd(normalized_feature_importance[i], normalized_feature_importance[j])
        similarity_matrix_xgboost[i, j] = emd_value
        similarity_matrix_xgboost[j, i] = emd_value

print(similarity_matrix_xgboost.shape)



# Plot the similarity matrix
plt.figure(figsize=(30, 10))
plt.imshow(similarity_matrix_xgboost, cmap="YlGnBu", interpolation='nearest', aspect='auto')

# Add text annotations for each grid
for i in range(len(region_names)):
    for j in range(len(region_names)):
        plt.text(j, i, f'{similarity_matrix_xgboost[i, j]:.5f}', ha='center', va='center', color='black')

plt.colorbar()
plt.title('Similarity Matrix based on Earth Mover\'s Distance (EMD) between Normalized XGBoost Feature Importance Scores')
plt.xticks(np.arange(len(region_names)), region_names, rotation=90)
plt.yticks(np.arange(len(region_names)), region_names)
plt.show()





from sklearn.metrics.pairwise import pairwise_distances
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans

# Normalize feature importance scores
normalized_feature_importance = normalize_feature_importance(feature_importance_scores)

# Calculate pairwise EMD distances
emd_distances = pairwise_distances(normalized_feature_importance, metric=calculate_emd)

from scipy.cluster.hierarchy import linkage, fcluster

# Calculate linkage matrix using EMD distances
linkage_matrix = linkage(emd_distances, method='average')  # You can adjust the method as desired

# Perform agglomerative clustering
num_clusters = 3  # Set the number of clusters as desired
cluster_labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')

# Print cluster labels for each region
for region, label in zip(region_names, cluster_labels):
    print(f"Region: {region}, Cluster: {label}")



