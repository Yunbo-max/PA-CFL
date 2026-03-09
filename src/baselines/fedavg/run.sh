#!/bin/bash
set -e
cd "$(dirname "$0")/../../.."

CONFIG="configs/fedavg.json"
DATA="data/processed/datasets.pkl"
REGIONS=(2 5 6 7 9 12 14 17 22)

python -m src.baselines.fedavg.server &
sleep 3

for i in "${REGIONS[@]}"; do
    echo "Starting FedAvg client $i"
    python -m src.baselines.fedavg.client \
        --client_number="$i" --config="$CONFIG" --data_path="$DATA" \
        > "logs/fedavg_client_$i.log" 2>&1 &
done

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
wait
