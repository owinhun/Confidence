#!/usr/bin/env bash
# The true-claim half of the ladder swap, which wave9b lost.
#
# wave9b queued the true-claim rungs as conditions named `v_vlow_T`, `n_085_T` and so on. Those
# are filename suffixes, not conditions: the scripts take `--claim true` and append `_T`
# themselves. Every `_T` job therefore died at argument parsing while its false-claim twin ran
# fine, leaving the compression swing half-measured in the swapped order.
set -uo pipefail
cd /workspace/svhalluc/claude_track2
Q=/root/.claude/jobs/65f6dcfb/tmp/w9c.txt
LOCK=/root/.claude/jobs/65f6dcfb/tmp/w9c.lock
LOG=outputs/wave9_log.txt
: > "$Q"
for spec in "nm_qwen2audio new" "nm_af3 af3" "nm_nemotron nemo"; do
  set -- $spec
  for c in v_vlow v_high n_015 n_085; do echo "$1 $2 $c" >> "$Q"; done
done
echo "[queue-c] $(wc -l < "$Q") true-claim swap jobs" | tee -a "$LOG"
take() { flock "$LOCK" bash -c "l=\$(head -1 $Q); [ -n \"\$l\" ] && sed -i 1d $Q; echo \"\$l\""; }
worker() {
  local gpu=$1
  while :; do
    local job; job=$(take); [ -z "$job" ] && break
    set -- $job; local pfx=$1 kind=$2 cond=$3
    local script need=30000 extra="--swap_options --claim true"
    case $kind in
      new)  script=scripts/eval_newmodels_v1.py; extra="$extra --model ${pfx#nm_}" ;;
      af3)  script=scripts/run_af3_v1.sh ;;
      nemo) script=scripts/run_nemotron_v1.sh; need=65000 ;;
    esac
    local out="outputs/${pfx}_sound_${cond}_T_swap_v1.jsonl"
    if [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ]; then
      echo "[skip] $(basename $out)" | tee -a "$LOG"; continue
    fi
    while :; do
      local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu")
      [ "$free" -ge "$need" ] && break
      sleep 45
    done
    echo "[gpu$gpu] $(date +%H:%M) start $pfx $cond claim=true swap" | tee -a "$LOG"
    if [[ $script == *.sh ]]; then
      CUDA_VISIBLE_DEVICES=$gpu bash "$script" --attr sound --cond "$cond" $extra \
        >> "outputs/w9c_${pfx}_${cond}.out" 2>&1 || true
    else
      CUDA_VISIBLE_DEVICES=$gpu python3 "$script" --attr sound --cond "$cond" $extra \
        >> "outputs/w9c_${pfx}_${cond}.out" 2>&1 || true
    fi
    echo "[gpu$gpu] $(date +%H:%M) done  $pfx $cond claim=true swap rows=$( [ -f "$out" ] && wc -l < "$out" || echo 0)" | tee -a "$LOG"
  done
}
for g in 0 2 3; do worker "$g" & done
wait
echo "[wave9c] all done $(date)" | tee -a "$LOG"
