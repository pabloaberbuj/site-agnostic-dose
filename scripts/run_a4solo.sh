#!/usr/bin/env bash
# HO#9: entrena dist-solo (A0 + PSDM PTV_High) y evalúa en val (test intacto).  Uso: bash scripts/run_a4solo.sh
set -u
PY=.venv/Scripts/python.exe
name=exp_a4_002_unet2d_distsolo; cfg="configs/${name}.yaml"
mkdir -p logs
$PY scripts/train_a0.py --config "$cfg" 2>&1 | tr '\r' '\n' > "logs/${name}_train.log"
ckpt=$(grep '^best:' "logs/${name}_train.log" | sed -E 's/^best: (.*\.ckpt) .*/\1/')
echo "[${name}] best ckpt: ${ckpt}"
[ -n "$ckpt" ] || { echo "sin best ckpt"; exit 1; }
$PY scripts/eval_a0.py net --config "$cfg" --ckpt "$ckpt" --split val --out-dir "results/${name}" > "logs/${name}_eval.log" 2>&1
echo "dist-solo train+eval completo"
