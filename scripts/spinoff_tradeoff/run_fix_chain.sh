#!/usr/bin/env bash
# Rehace Etapa 1 (s3-s5) y Etapa 1b (s7-s8) tras corregir el R² (recorte de z en descriptores casi constantes).
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
PY=.venv/Scripts/python.exe
rm -f logs/spinoff_fix.done
$PY -u scripts/spinoff_tradeoff/s3_residual.py Prostata Pelvis CyC > logs/spinoff_fix_s3.log 2>&1 && \
$PY -u scripts/spinoff_tradeoff/s4_replica.py > logs/spinoff_fix_s4.log 2>&1 && \
$PY -u scripts/spinoff_tradeoff/s5_report.py > logs/spinoff_fix_s5.log 2>&1 && \
$PY -u scripts/spinoff_tradeoff/s7_report_1b.py > logs/spinoff_fix_s7.log 2>&1 && \
$PY -u scripts/spinoff_tradeoff/s8_robust_1b.py > logs/spinoff_fix_s8.log 2>&1 && date > logs/spinoff_fix.done
