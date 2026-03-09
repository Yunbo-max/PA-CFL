# -*- coding: utf-8 -*-
# PA-CFL Full Pipeline:
#   1. Compute feature importance per client (XGBoost)
#   2. Apply differential privacy (Laplace noise)
#   3. Cluster clients into bubbles (Agglomerative + DBI)
#   4. Launch FL training per bubble
#
# Usage: python -m src.pacfl.run_pipeline --data_path data/processed/datasets.pkl --epsilon 10

import argparse
import json
import os
import pickle
import subprocess
import time

from src.data.preprocess import REGION_MAP, ACTIVE_REGIONS
from src.pacfl.clustering.feature_importance import compute_all_feature_importances
from src.pacfl.clustering.differential_privacy import apply_differential_privacy
from src.pacfl.clustering.clustering import perform_clustering, identify_attackers


def main():
    parser = argparse.ArgumentParser(description='PA-CFL Pipeline')
    parser.add_argument('--data_path', type=str, default='data/processed/datasets.pkl')
    parser.add_argument('--epsilon', type=float, default=10.0,
                        help='Privacy budget (0.1=high privacy, 10=low privacy)')
    parser.add_argument('--n_clusters', type=int, default=None,
                        help='Number of clusters (None=auto via DBI)')
    parser.add_argument('--num_rounds', type=int, default=100)
    parser.add_argument('--base_port', type=int, default=8090)
    args = parser.parse_args()

    # Load data
    with open(args.data_path, 'rb') as f:
        datasets = pickle.load(f)

    region_ids = [(REGION_MAP[r], r) for r in ACTIVE_REGIONS if REGION_MAP[r] in datasets]

    # Step 1: Feature Importance
    print("=" * 60)
    print("Step 1: Computing feature importance (XGBoost)...")
    importance_matrix, client_names = compute_all_feature_importances(datasets, region_ids)
    print(f"  Computed importance for {len(client_names)} clients")

    # Step 2: Differential Privacy
    print("=" * 60)
    print(f"Step 2: Applying differential privacy (epsilon={args.epsilon})...")
    noisy_matrix = apply_differential_privacy(importance_matrix, epsilon=args.epsilon)
    print(f"  Added Laplace noise to feature importance scores")

    # Step 3: Clustering
    print("=" * 60)
    print("Step 3: Clustering clients into bubbles...")
    labels, n_clusters, bubbles = perform_clustering(noisy_matrix, args.n_clusters)
    attackers, safe_bubbles = identify_attackers(bubbles)

    print(f"  Optimal clusters: {n_clusters}")
    for bid, clients in safe_bubbles.items():
        names = [client_names[c] for c in clients]
        print(f"  Bubble {bid}: {names}")
    if attackers:
        attacker_names = [client_names[a] for a in attackers]
        print(f"  Flagged as potential attackers (isolated): {attacker_names}")

    # Step 4: Launch FL per bubble
    print("=" * 60)
    print("Step 4: Launching federated learning per bubble...")

    # Map client names back to region IDs
    name_to_id = {name: rid for name, rid in region_ids}

    os.makedirs("logs", exist_ok=True)
    processes = []

    for bubble_id, client_indices in safe_bubbles.items():
        port = args.base_port + bubble_id
        n_clients = len(client_indices)
        client_region_ids = [name_to_id[client_names[c]] for c in client_indices]

        # Write bubble config
        bubble_config = {
            "project_name": f"PA-CFL_bubble_{bubble_id}",
            "server_address": f"127.0.0.1:{port}"
        }
        config_path = f"configs/pacfl_bubble_{bubble_id}.json"
        with open(config_path, 'w') as f:
            json.dump(bubble_config, f, indent=2)

        # Start server
        server_cmd = [
            "python", "-m", "src.pacfl.server",
            f"--bubble_id={bubble_id}",
            f"--port={port}",
            f"--min_clients={n_clients}",
            f"--num_rounds={args.num_rounds}",
        ]
        print(f"  Starting server for bubble {bubble_id} on port {port}")
        sp = subprocess.Popen(server_cmd,
                              stdout=open(f"logs/pacfl_server_{bubble_id}.log", "w"),
                              stderr=subprocess.STDOUT)
        processes.append(sp)

        time.sleep(3)  # Wait for server to start

        # Start clients
        for rid in client_region_ids:
            client_cmd = [
                "python", "-m", "src.pacfl.client",
                f"--client_number={rid}",
                f"--config={config_path}",
                f"--data_path={args.data_path}",
                f"--bubble_id={bubble_id}",
            ]
            print(f"  Starting client {REGION_MAP[rid]} (region {rid}) in bubble {bubble_id}")
            cp = subprocess.Popen(client_cmd,
                                  stdout=open(f"logs/pacfl_client_{rid}.log", "w"),
                                  stderr=subprocess.STDOUT)
            processes.append(cp)

    print("=" * 60)
    print(f"All processes launched. Waiting for completion...")
    for p in processes:
        p.wait()
    print("Done.")


if __name__ == '__main__':
    main()
