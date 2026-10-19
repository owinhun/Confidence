#!/usr/bin/env bash
# AF3 and Nemotron halves of the corrected confidence cascade, through their transformers-5.x
# wrapper. Re-run because the first pass forwarded a confidence built about the wrong quantity:
# the decision threshold was 0.876 but the confidence was max(s,1-s), i.e. measured against 0.5,
# so on a third of the clips the declared level described belief in the opposite of the claim.
# cascade_mono is not re-run -- with the confidence built correctly the bands are monotone, so
# the monotone repair is the identity map.
set -uo pipefail
cd /workspace/svhalluc/claude_track2
LOG=outputs/conf_rerun_log.txt
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
    >> "outputs/confrr_${model}_${cond}${tag}.out" 2>&1 || true
  echo "[gpu$gpu] $(date +%H:%M) done  $model $cond $order rows=$( [ -f "$out" ] && wc -l < "$out" || echo 0)" | tee -a "$LOG"
}
for o in none swap; do
  for c in cascade_conf cascade_pct; do go af3 "$c" "$o" 30000 scripts/run_af3_v1.sh; done
done
for o in none swap; do
  for c in cascade_bare cascade_conf cascade_pct; do go nemotron "$c" "$o" 65000 scripts/run_nemotron_v1.sh; done
done
echo "[conf-rerun] all done $(date)" | tee -a "$LOG"
