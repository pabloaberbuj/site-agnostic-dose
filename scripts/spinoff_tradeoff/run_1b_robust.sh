#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
rm -f logs/spinoff_1b_robust.done
.venv/Scripts/python.exe -u scripts/spinoff_tradeoff/s8_robust_1b.py > logs/spinoff_1b_robust.log 2>&1 && date > logs/spinoff_1b_robust.done
