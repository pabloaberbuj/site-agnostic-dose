#!/usr/bin/env bash
# Extracción A+B (inferencia A4 ep113 sobre los 432). Reanudable (cache por paciente). Convive con un entrenamiento en GPU.
cd "$(dirname "$0")/../.."
mkdir -p logs
for i in $(seq 1 15); do
  .venv/Scripts/python.exe scripts/spinoff_tradeoff/s1_extract.py 2>&1 | tr '\r' '\n' >> logs/spinoff_s1.log
  rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] && { date > logs/spinoff_s1.done; exit 0; }
  echo "[retry] intento $i exit $rc $(date)" >> logs/spinoff_s1.log; sleep 60
done
exit 1
