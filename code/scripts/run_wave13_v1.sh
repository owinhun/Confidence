#!/usr/bin/env bash
# Complete the Qwen3/noise battery so every "primary cells" count in the paper names one set.
#
# wave12 put the five-rung ladder and the false-claim control fields on this cell, which was
# enough to draw it in Figure 1. The field-name swing needs the true-claim branch of both the
# epistemic and the non-epistemic field as well, and those were never run here. Without them
# Figure 1 would show seven primary cells while the swing sentence said six -- the same
# two-different-sevens problem wave12 set out to remove, moved to a new place.
#
# GPUs 0/1 only: the machine is busy and 2/3 were handed back.
set -uo pipefail
cd /workspace/svhalluc/claude_track2
Q=/root/.claude/jobs/65f6dcfb/tmp/w13.txt
LOCK=/root/.claude/jobs/65f6dcfb/tmp/w13.lock
LOG=outputs/wave13_log.txt
: > "$Q"
for c in true_high true_low priority_true_high priority_true_low; do
  for o in none swap; do echo "$c $o" >> "$Q"; done
done
echo "[queue] $(wc -l < "$Q") jobs $(date)" | tee -a "$LOG"

take() { flock "$LOCK" bash -c "l=\$(head -1 $Q); [ -n \"\$l\" ] && sed -i 1d $Q; echo \"\$l\""; }

worker() {
  local gpu=$1
  while :; do
    local job; job=$(take); [ -z "$job" ] && break
    set -- $job; local cond=$1 order=$2 extra="" tag=""
    [ "$order" = swap ] && { extra="--swap_options"; tag="_swap"; }
    local out="outputs/s2q3_noise_${cond}${tag}_v1.jsonl"
    if [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ]; then
      echo "[skip] $(basename $out)" | tee -a "$LOG"; continue
    fi
    while :; do
      local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu")
      [ "$free" -ge 45000 ] && break
      sleep 45
    done
    echo "[gpu$gpu] $(date +%H:%M) start s2q3 noise $cond $order" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=$gpu python3 scripts/eval_stage2_q3_v1.py --attr noise --cond "$cond" $extra \
      >> "outputs/w13_s2q3_${cond}${tag}.out" 2>&1 || true
    echo "[gpu$gpu] $(date +%H:%M) done  s2q3 noise $cond $order rows=$( [ -f "$out" ] && wc -l < "$out" || echo 0)" | tee -a "$LOG"
  done
  echo "[gpu$gpu] empty" | tee -a "$LOG"
}
for g in 0 1; do worker "$g" & done
wait
echo "[wave13] all done $(date)" | tee -a "$LOG"
