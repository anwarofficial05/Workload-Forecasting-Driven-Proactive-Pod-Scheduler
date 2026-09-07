#!/usr/bin/env bash
set -e

echo "============================================================"
echo " Executing End-to-End Comparative Scheduling Experiment"
echo "============================================================"

echo "[1/3] Running comparative evaluation against baselines..."
python evaluation/evaluate.py

echo "[2/3] Generating 13 publication-quality comparison plots..."
python evaluation/plots.py

echo "[3/3] Inspecting output figures in evaluation/plots/..."
ls -lh evaluation/plots/

echo "============================================================"
echo " Experiment Complete! Results saved to evaluation/results/"
echo "============================================================"
