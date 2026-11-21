#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
rm -f logs/spinoff_1b_analysis.done
.venv/Scripts/python.exe -u scripts/spinoff_tradeoff/s7_report_1b.py > logs/spinoff_1b_analysis.log 2>&1 && date > logs/spinoff_1b_analysis.done
