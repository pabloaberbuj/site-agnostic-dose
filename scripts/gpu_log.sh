#!/usr/bin/env bash
# Log de GPU cada 60 s (temp, potencia, util, memoria, throttle) para diagnosticar cortes. Append a logs/gpu_log.csv.
cd "$(dirname "$0")/.."
[ -f logs/gpu_log.csv ] || echo "ts,temp_c,power_w,util_pct,mem_mib,clk_mhz,throttle_reasons" > logs/gpu_log.csv
while true; do
  echo "$(date '+%F %T'),$(nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used,clocks.sm,clocks_throttle_reasons.active --format=csv,noheader,nounits | tr -d ' \r')" >> logs/gpu_log.csv
  sleep 60
done
