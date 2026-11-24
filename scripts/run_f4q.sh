#!/usr/bin/env bash
# Fase 4 cuantilica: entrena los tags de TAGS en serie (from-scratch, --resume, reintentos). Solo train; eval/seleccion aparte.
#   Piloto:  TAGS="0p35"      powershell -File scripts/launch_task.ps1 -Name sa_f4q_pilot -Cmd "TAGS=0p35 bash scripts/run_f4q.sh > logs/run_f4q_pilot.out 2>&1"
#   Resto:   TAGS="0p30 0p40" (solo si el piloto pasa el gate)
set -u
cd "$(dirname "$0")/.."
source scripts/_lib_run.sh
PY=.venv/Scripts/python.exe
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
mkdir -p logs
TRAIN_W="${TRAIN_W:-2}"
train_once () {
  $PY scripts/train_a0.py --config "$1" --resume --min-free-gib "${MIN_FREE_GIB:-3.0}" --cap-gib 11.0 --workers "$TRAIN_W" 2>&1 | tr '\r' '\n' >> "$2"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 4 ] && [ "$TRAIN_W" -gt 0 ]; then TRAIN_W=0; echo "[workers] corte por RAM: reintentos con 0 workers"; fi
  return "$rc"
}
echo "[f4q] inicio $(date '+%F %T') TAGS=$TAGS"
for TAG in $TAGS; do
  CFG=$(ls configs/exp_f4q_*_pinball_tau$TAG.yaml); NAME=$(basename "$CFG" .yaml); LOG=logs/${NAME}_train.log
  [ -f logs/$NAME.done ] && continue
  wait_gpu_free || exit 1
  retry_cmd 30 train_once "$CFG" "$LOG" || { echo "[$NAME] train no terminó"; exit 1; }
  date > logs/$NAME.done
  echo "[$NAME] completo $(date '+%F %T')"
done
echo "[f4q] fin $(date '+%F %T')"
