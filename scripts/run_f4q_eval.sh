#!/usr/bin/env bash
# eval_a0 (metricas de la vara: PTV-alta 80-110, etc.) sobre los 5 ckpts cuantilicos. GPU libre.
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
for p in "exp_f4q_001_pinball_tau0p35 96" "exp_f4q_002_pinball_tau0p30 100" "exp_f4q_002_pinball_tau0p30 110" "exp_f4q_003_pinball_tau0p40 100" "exp_f4q_003_pinball_tau0p40 112"; do
  set -- $p
  [ -f results/$1/ckpt_ep$2/summary.json ] && continue
  $PY scripts/eval_a0.py net --config configs/$1.yaml --ckpt checkpoints/$1/epoch=$(printf %03d $2).ckpt --split val --out-dir results/$1/ckpt_ep$2 > logs/eval_$1_ep$2.log 2>&1 || echo "[$1 ep$2] fallo"
done
date > logs/f4q_eval.done
