#!/usr/bin/env bash
# Re-eval de exp_a4_003 con el harness vigente (pred*BODY, SSIM_BODY) = vara oficial de Fase 4. SOLO val, sin entrenar.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
N=exp_a4_003_unet2d_psdm_c176x256
O=results/${N}_vara_f4
for ep in 100 102 113; do
  [ -f $O/ckpt_ep$ep/summary.json ] && continue
  $PY scripts/eval_a0.py net --config configs/$N.yaml --ckpt checkpoints/$N/epoch=$ep.ckpt --split val --out-dir $O/ckpt_ep$ep > logs/vara_f4_ep$ep.log 2>&1 || { echo "eval ep$ep fallo"; exit 1; }
done
echo fin_evals
