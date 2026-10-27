#!/usr/bin/env bash
# HO#10: entrena A5 (A4 + densidad) y evalúa en val (test intacto).  Uso: bash scripts/run_a5.sh
set -u
PY=.venv/Scripts/python.exe
name=exp_a5_001_unet2d_density; cfg="configs/${name}.yaml"
mkdir -p logs
$PY scripts/train_a0.py --config "$cfg" 2>&1 | tr '\r' '\n' > "logs/${name}_train.log"
ckpt=$(grep '^best:' "logs/${name}_train.log" | sed -E 's/^best: (.*\.ckpt) .*/\1/')
echo "[${name}] best ckpt: ${ckpt}"
[ -n "$ckpt" ] || { echo "sin best ckpt"; exit 1; }
$PY scripts/eval_a0.py net --config "$cfg" --ckpt "$ckpt" --split val --out-dir "results/${name}" > "logs/${name}_eval.log" 2>&1
echo "A5 train+eval completo"
