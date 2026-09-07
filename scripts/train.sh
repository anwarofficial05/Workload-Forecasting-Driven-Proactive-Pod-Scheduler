#!/usr/bin/env bash
set -e

echo "============================================================"
echo " Running LSTM and XGBoost Training Pipeline"
echo "============================================================"

python module3_workload_prediction/train.py \
  --config config/config.yaml \
  --epochs 15 \
  --batch-size 32 \
  --samples 2000

echo "Training completed successfully. Models saved to models/"
