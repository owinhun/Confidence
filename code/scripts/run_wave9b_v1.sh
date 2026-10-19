#!/usr/bin/env bash
# Swapped-order runs for the block-format finding, plus the ladder rungs for the three models
# whose ladder lives in eval_newmodels (which supports the swap).
#
# eval_confscale_v1.py, which owns the Qwen2.5 and Qwen3 ladders, has no --swap_options; adding
# it is a separate change. The block-format result is the one v10 leads with, so it goes first.
set -uo pipefail
cd /workspace/svhalluc/claude_track2
Q=/root/.claude/jobs/65f6dcfb/tmp/w9b.txt
LOCK=/root/.claude/jobs/65f6dcfb/tmp/w9b.lock
LOG=outputs/wave9_log.txt
: > "$Q"
for spec in "s2 fact" "s2q3 q3" "nm_qwen2audio new" "nm_af3 af3" "nm_nemotron nemo"; do
  set -- $spec
  echo "$1 $2 bare_true" >> "$Q"
  echo "$1 $2 bare_false" >> "$Q"
done
for spec in "nm_qwen2audio new" "nm_af3 af3" "nm_nemotron nemo"; do
  set -- $spec
  for c in v_vlow v_vlow_T v_high v_high_T n_015 n_015_T n_085 n_085_T; do
    echo "$1 $2 $c" >> "$Q"
  done
done
echo "[queue-b] $(wc -l < "$Q") jobs" | tee -a "$LOG"
take() { flock "$LOCK" bash -c "l=\$(head -1 $Q); [ -n \"\$l\" ] && sed -i 1d $Q; echo \"\$l\""; }
worker() {
  local gpu=$1
  while :; do
    local job; job=$(take); [ -z "$job" ] && break
    set -- $job; local pfx=$1 kind=$2 cond=$3
    local script need=30000 extra="--swap_options"
    case $kind in
      fact) script=scripts/eval_stage2_factorial_v1.py ;;
      q3)   script=scripts/eval_stage2_q3_v1.py; need=45000 ;;
      new)  script=scripts/eval_newmodels_v1.py; extra="$extra --model ${pfx#nm_}" ;;
      af3)  script=scripts/run_af3_v1.sh ;;
      nemo) script=scripts/run_nemotron_v1.sh; need=65000 ;;
    esac
    local out="outputs/${pfx}_sound_${cond}_swap_v1.jsonl"
    if [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ]; then
      echo "[skip] $(basename $out)" | tee -a "$LOG"; continue
    fi
    while :; do
      local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu")
      [ "$free" -ge "$need" ] && break
      sleep 45
    done
    echo "[gpu$gpu] $(date +%H:%M) start $pfx $cond swap" | tee -a "$LOG"
    if [[ $script == *.sh ]]; then
      CUDA_VISIBLE_DEVICES=$gpu bash "$script" --attr sound --cond "$cond" $extra \
        >> "outputs/w9_${pfx}_${cond}.out" 2>&1 || true
    else
      CUDA_VISIBLE_DEVICES=$gpu python3 "$script" --attr sound --cond "$cond" $extra \
        >> "outputs/w9_${pfx}_${cond}.out" 2>&1 || true
    fi
    echo "[gpu$gpu] $(date +%H:%M) done  $pfx $cond swap rows=$( [ -f "$out" ] && wc -l < "$out" || echo 0)" | tee -a "$LOG"
  done
  echo "[gpu$gpu] empty" | tee -a "$LOG"
}
for g in 0 1 2 3; do worker "$g" & done
wait
echo "[wave9b] all done $(date)" | tee -a "$LOG"
