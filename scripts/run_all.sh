#!/bin/bash
# Run all experiments: baselines + PA-CFL
# Usage: bash scripts/run_all.sh
set -e
cd "$(dirname "$0")/.."

mkdir -p logs

echo "============================================"
echo "Step 0: Preparing data..."
echo "============================================"
python scripts/prepare_data.py

echo ""
echo "============================================"
echo "Step 1: Local Learning Baselines"
echo "============================================"
for model in transformer lstm gru cnn mlp; do
    echo "--- Running $model local learning ---"
    python -m src.baselines.local_learning --model "$model" \
        --data_path data/processed/datasets.pkl > "logs/local_${model}.log" 2>&1
    echo "  Done: $model"
done

echo ""
echo "============================================"
echo "Step 2: FedAvg Baseline"
echo "============================================"
bash src/baselines/fedavg/run.sh

echo ""
echo "============================================"
echo "Step 3: FedProx Baseline (mu=0.01)"
echo "============================================"
bash src/baselines/fedprox/run.sh

echo ""
echo "============================================"
echo "Step 4: IFCA Baseline (K=3)"
echo "============================================"
bash src/baselines/ifca/run.sh

echo ""
echo "============================================"
echo "Step 5: PA-CFL (Ours)"
echo "============================================"
bash src/pacfl/run.sh

echo ""
echo "============================================"
echo "All experiments complete."
echo "Check WandB for results."
echo "============================================"
