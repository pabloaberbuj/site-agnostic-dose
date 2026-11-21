#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe
rm -f logs/spinoff_1b_extract.done
$PY -u scripts/spinoff_tradeoff/s1b_extract.py --phase real > logs/spinoff_1b_real.log 2>&1 && \
CUDA_VISIBLE_DEVICES= $PY -u scripts/spinoff_tradeoff/s1b_extract.py --phase pred --threads 6 > logs/spinoff_1b_pred.log 2>&1 && date > logs/spinoff_1b_extract.done
