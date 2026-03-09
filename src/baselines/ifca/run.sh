#!/bin/bash
set -e
cd "$(dirname "$0")/../../.."

CONFIG="configs/ifca.json"
DATA="data/processed/datasets.pkl"
K=${1:-3}
REGIONS=(2 5 6 7 9 12 14 17 22)

python -m src.baselines.ifca.server &
sleep 3

for i in "${REGIONS[@]}"; do
    echo "Starting IFCA client $i (K=$K)"
    python -m src.baselines.ifca.client \
        --client_number="$i" --config="$CONFIG" --data_path="$DATA" --num_clusters="$K" \
        > "logs/ifca_client_$i.log" 2>&1 &
done

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
wait
