#!/usr/bin/env bash
# HO#5: entrena A0 con seeds 43/44/45 en secuencia (un solo proceso -> sin contienda de GPU) y evalúa cada uno en val.
# Uso (desde la raíz del repo):  bash scripts/run_ho5_seeds.sh
set -u
PY=.venv/Scripts/python.exe
for s in 43 44 45; do
  n=$((s-41)); name="exp_a0_00${n}_unet2d_seed${s}"; cfg="configs/${name}.yaml"
  mkdir -p logs
  if [ -f "results/${name}/summary.json" ]; then echo "[skip] ${name} ya evaluado"; continue; fi
  $PY scripts/train_a0.py --config "$cfg" 2>&1 | tr '\r' '\n' > "logs/${name}_train.log"
  ckpt=$(grep '^best:' "logs/${name}_train.log" | sed -E 's/^best: (.*\.ckpt) .*/\1/')
  echo "[${name}] best ckpt: ${ckpt}"
  [ -n "$ckpt" ] || { echo "[${name}] sin best ckpt, abortando"; exit 1; }
  $PY scripts/eval_a0.py net --config "$cfg" --ckpt "$ckpt" --split val --out-dir "results/${name}" \
    > "logs/${name}_eval.log" 2>&1
done
echo "HO5 train+eval completo"
