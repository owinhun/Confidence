#!/usr/bin/env bash
# Claim-only controlled branches for the two transformers-5.x models.
set -uo pipefail
cd /workspace/svhalluc/claude_track2
LOG=outputs/detclaim_log.txt
go() {
  local model=$1 cond=$2 order=$3 need=$4 wrapper=$5
  local tag=""; [ "$order" = swap ] && tag="_swap"
  local out="outputs/nm_${model}_sound_${cond}${tag}_v1.jsonl"
  if [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ]; then echo "[skip] $out" | tee -a "$LOG"; return; fi
  local gpu=""
  while [ -z "$gpu" ]; do
    for g in 0 1 2 3; do
      local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g")
      if [ "$free" -ge "$need" ]; then gpu=$g; break; fi
    done
    [ -z "$gpu" ] && sleep 120
  done
  local extra=""; [ "$order" = swap ] && extra="--swap_options"
  echo "[gpu$gpu] $(date +%H:%M) start $model $cond $order" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$gpu bash "$wrapper" --attr sound --cond "$cond" $extra \
    >> "outputs/dc_${model}_${cond}${tag}.out" 2>&1 || true
  echo "[gpu$gpu] $(date +%H:%M) done  $model $cond $order rows=$( [ -f "$out" ] && wc -l < "$out" || echo 0)" | tee -a "$LOG"
}
for o in none swap; do
  for c in detclaim_true detclaim_false; do go af3 "$c" "$o" 30000 scripts/run_af3_v1.sh; done
done
for o in none swap; do
  for c in detclaim_true detclaim_false; do go nemotron "$c" "$o" 65000 scripts/run_nemotron_v1.sh; done
done
echo "[detclaim] all done $(date)" | tee -a "$LOG"
