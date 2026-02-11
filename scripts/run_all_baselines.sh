#!/bin/bash

echo "=========================================="
echo "EJECUTANDO BASELINES"
echo "=========================================="

echo ""
echo "→ Baseline 1: None (30 épocas)"
python /app/scripts/run_baseline_none.py

echo ""
echo "→ Baseline 2: Ben Graham (30 épocas)"
python /app/scripts/run_baseline_ben_graham.py

echo ""
echo "=========================================="
echo "✓ TODOS LOS BASELINES COMPLETADOS"
echo "=========================================="
echo ""
echo "Para ver resultados:"
echo "mlflow ui --backend-store-uri /app/experiments/mlruns --host 0.0.0.0 --port 5000"