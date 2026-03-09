#!/bin/bash
# PA-CFL: Run the full pipeline
# Usage: bash src/pacfl/run.sh [--epsilon 10] [--n_clusters 3]
set -e
cd "$(dirname "$0")/../.."

EPSILON=${1:-10}
mkdir -p logs

python -m src.pacfl.run_pipeline \
    --data_path data/processed/datasets.pkl \
    --epsilon "$EPSILON"
