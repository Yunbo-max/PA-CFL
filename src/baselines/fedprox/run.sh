#!/bin/bash
set -e
cd "$(dirname "$0")/../../.."

CONFIG="configs/fedprox.json"
DATA="data/processed/datasets.pkl"
MU=${1:-0.01}
REGIONS=(2 5 6 7 9 12 14 17 22)

python -m src.baselines.fedprox.server &
sleep 3

for i in "${REGIONS[@]}"; do
    echo "Starting FedProx client $i (mu=$MU)"
    python -m src.baselines.fedprox.client \
        --client_number="$i" --config="$CONFIG" --data_path="$DATA" --mu="$MU" \
        > "logs/fedprox_client_$i.log" 2>&1 &
done

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
wait
