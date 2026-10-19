#!/usr/bin/env bash
# Swapped-order cascade for the two models that need transformers 5.x. The headline cascade
# table was reported in one answer order while every other table in the paper is order-averaged;
# Sec. 6 shows magnitudes are order-dependent, so a single-order cascade table is exactly the
# inconsistency a reviewer will find. These runs let the whole table be averaged like the rest.
set -uo pipefail
cd /workspace/svhalluc/claude_track2
LOG=outputs/cascade_swap_log.txt
run() {
  local model=$1 cond=$2 need=$3 wrapper=$4
  local out="outputs/nm_${model}_sound_${cond}_swap_v1.jsonl"
  if [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ]; then echo "[skip] $out" | tee -a "$LOG"; return; fi
  local gpu=""
  while [ -z "$gpu" ]; do
    for g in 0 1 2 3; do
      local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g")
      if [ "$free" -ge "$need" ]; then gpu=$g; break; fi
    done
    [ -z "$gpu" ] && sleep 120
  done
  echo "[gpu$gpu] $(date +%H:%M) start $model $cond swap" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$gpu bash "$wrapper" --attr sound --cond "$cond" --swap_options \
    >> "outputs/cswap_${model}_${cond}.out" 2>&1 || true
  echo "[gpu$gpu] $(date +%H:%M) done  $model $cond swap rows=$( [ -f "$out" ] && wc -l < "$out" || echo 0)" | tee -a "$LOG"
}
for c in cascade_bare cascade_conf cascade_mono cascade_pct; do
  run af3 "$c" 30000 scripts/run_af3_v1.sh
done
for c in cascade_bare cascade_conf cascade_mono cascade_pct; do
  run nemotron "$c" 65000 scripts/run_nemotron_v1.sh
done
echo "[cascade-swap] all done $(date)" | tee -a "$LOG"
