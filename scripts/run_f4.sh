#!/usr/bin/env bash
# Fase 4, loss #1 (momentos serie/paralelo + MAE): sweep λ in {0.1, 0.3, 1.0}, secuencial, desacoplado y reanudable.
# Por λ: train (--resume, reintentos) -> eval val de los top-3 por val/loss (results/<exp>/ckpt_ep<N>) -> selección (menor criterio
# DVH sobre top-3; reporta ambos) -> d2 axial del elegido. Pasos terminados dejan logs/<exp>.done / .eval.done y se saltean.
#   powershell -File scripts/launch_task.ps1 -Name sa_f4 -Cmd "bash scripts/run_f4.sh > logs/run_f4.out 2>&1"
set -u
cd "$(dirname "$0")/.."
source scripts/_lib_run.sh
PY=.venv/Scripts/python.exe
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
mkdir -p logs
TRAIN_W="${TRAIN_W:-4}"   # VM apagada: ~22 GiB libres, 24 hilos. Si el watchdog corta (exit 4) los reintentos bajan a 0
train_once () {
  $PY scripts/train_a0.py --config "$1" --resume --min-free-gib "${MIN_FREE_GIB:-3.0}" --cap-gib 11.0 --workers "$TRAIN_W" 2>&1 | tr '\r' '\n' >> "$2"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 4 ] && [ "$TRAIN_W" -gt 0 ]; then TRAIN_W=0; echo "[workers] corte por RAM: los reintentos siguen con 0 workers"; fi
  return "$rc"
}
echo "[f4] inicio $(date '+%F %T')"
for TAG in 0p3 0p1 1p0; do
  NAME=exp_f4_001_dvhmoment_sp_lambda$TAG; CFG=configs/$NAME.yaml; LOG=logs/${NAME}_train.log
  if [ ! -f logs/${NAME}.done ]; then
    wait_gpu_free || exit 1
    retry_cmd 30 train_once "$CFG" "$LOG" || { echo "[$NAME] train no terminó"; exit 1; }
    date > logs/${NAME}.done
  fi
  if [ ! -f logs/${NAME}.eval.done ]; then
    for ck in checkpoints/$NAME/epoch=*.ckpt; do
      ep=$(basename "$ck" .ckpt | sed 's/epoch=//')
      [ -f results/$NAME/ckpt_ep$ep/summary.json ] && continue
      retry_cmd 3 $PY scripts/eval_a0.py net --config "$CFG" --ckpt "$ck" --split val --out-dir results/$NAME/ckpt_ep$ep > logs/eval_${NAME}_ep$ep.log 2>&1 || { echo "[$NAME] eval ep$ep falló"; exit 1; }
    done
    $PY scripts/f4_seleccion.py $NAME > logs/${NAME}_seleccion.log 2>&1 || { echo "[$NAME] selección falló"; exit 1; }
    cat logs/${NAME}_seleccion.log
    CHOSEN=$(grep -a '^CHOSEN=' logs/${NAME}_seleccion.log | sed 's/^CHOSEN=//')
    $PY scripts/d2_coherencia_axial.py --config "$CFG" --ckpt "$CHOSEN" > logs/${NAME}_d2.log 2>&1 || echo "[$NAME] d2 falló (no bloqueante)"
    date > logs/${NAME}.eval.done
  fi
  echo "[$NAME] completo $(date '+%F %T')"
done
echo "[f4] completo $(date '+%F %T')"
