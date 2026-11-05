#!/usr/bin/env bash
# Fase 3 / eje B, B2: MedNeXt-2D. Corrida desacoplada y reanudable (Scheduled Task, ver scripts/launch_task.ps1):
#   train (189/28, --resume, reintentos) -> eval val de los top-3 ckpts por val/mae -> d2 (coherencia axial) del mejor.
# Solo val; NO toca test. Los pasos terminados dejan logs/<name>.done / .eval.done y se saltean al relanzar.
#   Uso: bash scripts/run_b2.sh
set -u
cd "$(dirname "$0")/.."
source scripts/_lib_run.sh
PY=.venv/Scripts/python.exe
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
mkdir -p logs
NAME=exp_b2_001_mednext2d_psdm
CFG=configs/$NAME.yaml
LOG=logs/${NAME}_train.log
TRAIN_W="${TRAIN_W:-2}"
echo "[b2] inicio $(date '+%F %T')"
train_once () {
  $PY scripts/train_a0.py --config "$CFG" --resume --min-free-gib 1.0 --cap-gib 11.0 --workers "$TRAIN_W" 2>&1 | tr '\r' '\n' >> "$LOG"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 4 ] && [ "$TRAIN_W" -gt 0 ]; then TRAIN_W=0; echo "[workers] corte por RAM: los reintentos siguen con 0 workers"; fi
  return "$rc"
}
if [ ! -f logs/${NAME}.done ]; then
  wait_gpu_free || exit 1
  retry_cmd 30 train_once || { echo "[b2] train no terminó"; exit 1; }
  date > logs/${NAME}.done
fi
# top-3 por val/mae: el mejor va a results/<name>, los otros a results/<name>/ckpt_epNNN (misma convención que a4_003 / b1)
BEST=$(grep -a '^best:' "$LOG" | tail -1 | sed -E 's/^best: (.*\.ckpt) .*/\1/')
echo "[b2] best ckpt: $BEST"; [ -n "$BEST" ] || exit 1
if [ ! -f logs/${NAME}.eval.done ]; then
  for ck in checkpoints/$NAME/epoch=*.ckpt; do
    ep=$(basename "$ck" .ckpt | sed 's/epoch=//')
    if [ "$(realpath "$ck")" = "$(realpath "$BEST")" ]; then out=results/$NAME; else out=results/$NAME/ckpt_ep$ep; fi
    retry_cmd 3 $PY scripts/eval_a0.py net --config "$CFG" --ckpt "$ck" --split val --out-dir "$out" > "logs/eval_${NAME}_ep$ep.log" 2>&1 || { echo "[b2] eval ep$ep falló"; exit 1; }
  done
  $PY scripts/d2_coherencia_axial.py --config "$CFG" --ckpt "$BEST" > logs/${NAME}_d2.log 2>&1 || echo "[b2] d2 falló (no bloqueante)"
  date > logs/${NAME}.eval.done
fi
echo "[b2] completo $(date '+%F %T')"
