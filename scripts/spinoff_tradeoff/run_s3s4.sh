#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
PY=.venv/Scripts/python.exe
$PY -u scripts/spinoff_tradeoff/s3_residual.py Prostata Pelvis CyC > logs/spinoff_s3.log 2>&1 && \
$PY -u scripts/spinoff_tradeoff/s4_replica.py > logs/spinoff_s4.log 2>&1 && \
$PY -u scripts/spinoff_tradeoff/s5_report.py > logs/spinoff_s5.log 2>&1 && date > logs/spinoff_s345.done
