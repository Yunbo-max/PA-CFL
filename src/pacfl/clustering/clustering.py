# -*- coding: utf-8 -*-
# Step 3 & 4: Clustering Analysis
# Agglomerative clustering on noisy feature importance using EMD distance.
# Optimal K selected via Davies-Bouldin Index.

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score


def compute_emd_distance_matrix(importance_matrix):
    """Compute pairwise Earth Mover's Distance matrix.

    Args:
        importance_matrix: np.ndarray (n_clients, n_features)

    Returns:
        distance_matrix: np.ndarray (n_clients, n_clients)
    """
    n = len(importance_matrix)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = wasserstein_distance(importance_matrix[i], importance_matrix[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    return dist_matrix


def find_optimal_clusters(importance_matrix, max_k=None):
    """Find optimal number of clusters using Davies-Bouldin Index.

    Args:
        importance_matrix: np.ndarray (n_clients, n_features)
        max_k: maximum number of clusters to try (default: n_clients - 1)

    Returns:
        optimal_k: int
        db_scores: dict mapping k -> DBI score
    """
    n = len(importance_matrix)
    if max_k is None:
        max_k = min(n - 1, 10)

    dist_matrix = compute_emd_distance_matrix(importance_matrix)
    db_scores = {}

    for k in range(2, max_k + 1):
        clustering = AgglomerativeClustering(
            n_clusters=k,
            metric='precomputed',
            linkage='average'
        )
        labels = clustering.fit_predict(dist_matrix)
        db_scores[k] = davies_bouldin_score(importance_matrix, labels)

    optimal_k = min(db_scores, key=db_scores.get)
    return optimal_k, db_scores


def perform_clustering(importance_matrix, n_clusters=None):
    """Perform agglomerative clustering.

    Args:
        importance_matrix: np.ndarray (n_clients, n_features)
        n_clusters: int or None (auto-select via DBI)

    Returns:
        labels: np.ndarray (n_clients,) cluster assignments
        n_clusters: int, number of clusters used
        bubbles: dict mapping cluster_id -> list of client indices
    """
    if n_clusters is None:
        n_clusters, _ = find_optimal_clusters(importance_matrix)

    dist_matrix = compute_emd_distance_matrix(importance_matrix)

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average'
    )
    labels = clustering.fit_predict(dist_matrix)

    # Group clients into bubbles
    bubbles = {}
    for idx, label in enumerate(labels):
        if label not in bubbles:
            bubbles[label] = []
        bubbles[label].append(idx)

    return labels, n_clusters, bubbles


def identify_attackers(bubbles):
    """Flag single-client bubbles as potential attackers.

    Args:
        bubbles: dict mapping cluster_id -> list of client indices

    Returns:
        attackers: list of client indices in single-client bubbles
        safe_bubbles: dict with only multi-client bubbles
    """
    attackers = []
    safe_bubbles = {}

    for cluster_id, clients in bubbles.items():
        if len(clients) == 1:
            attackers.extend(clients)
        else:
            safe_bubbles[cluster_id] = clients

    return attackers, safe_bubbles
