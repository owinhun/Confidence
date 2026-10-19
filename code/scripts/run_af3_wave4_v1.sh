#!/usr/bin/env bash
# The AF3 half of wave4 plus its readback check, run through the wrapper that injects
# /data4/pyenv_af3. Pulled out of the main queue because that runner invokes the eval scripts
# directly and AF3 is the one model whose transformers version the shared environment cannot
# supply.
set -uo pipefail
cd /workspace/svhalluc/claude_track2
LOG=outputs/af3_wave4_log.txt
GPU=${1:-1}

go() {
  local cond=$1 rb=$2
  local suf=""; [ -n "$rb" ] && suf="_readback"
  local out="outputs/nm_af3_sound_${cond}${suf}_v1.jsonl"
  if [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ]; then
    echo "[skip] $out" | tee -a "$LOG"; return
  fi
  while :; do
    local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")
    [ "$free" -ge 30000 ] && break
    sleep 120
  done
  echo "[gpu$GPU] $(date +%H:%M) start af3 $cond$suf" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU bash scripts/run_af3_v1.sh --attr sound --cond "$cond" $rb \
    >> "outputs/af3w_${cond}${suf}.out" 2>&1 || true
  echo "[gpu$GPU] $(date +%H:%M) done  af3 $cond$suf rows=$( [ -f "$out" ] && wc -l < "$out" || echo 0)" | tee -a "$LOG"
}

go cascade_mono ""
go cascade_pct ""
go anon_high --readback
go true_high --readback
echo "[af3] all done $(date)" | tee -a "$LOG"
