#!/usr/bin/env bash
# The last nine conditions the field-name swing needs on the three cells wave14 completed.
#
# wave14 satisfied the battery definition the cell-flow count uses (anon and priority at high and
# low, plus the priority true branch). The swing needs the *Confidence* true branch as well --
# true_high and true_low -- and those were never run on these three cells. This is the same trap
# that caught wave12: a cell is only "added" once every statistic it appears in can be computed,
# not once one definition of complete is satisfied. Checked here against the swing's own
# requirement list rather than against the battery definition.
set -uo pipefail
cd /workspace/svhalluc/claude_track2
Q=/root/.claude/jobs/65f6dcfb/tmp/w15.txt
LOCK=/root/.claude/jobs/65f6dcfb/tmp/w15.lock
LOG=outputs/wave15_log.txt
cp /root/.claude/jobs/65f6dcfb/tmp/w15_jobs.txt "$Q"
echo "[queue] $(wc -l < "$Q") jobs $(date)" | tee -a "$LOG"

take() { flock "$LOCK" bash -c "l=\$(head -1 $Q); [ -n \"\$l\" ] && sed -i 1d $Q; echo \"\$l\""; }

worker() {
  local gpu=$1
  while :; do
    local job; job=$(take); [ -z "$job" ] && break
    set -- $job
    local pfx=$1 kind=$2 attr=$3 cond=$4 order=$5
    local script need=30000 extra="" tag=""
    case $kind in
      fact) script=scripts/eval_stage2_factorial_v1.py ;;
      q3)   script=scripts/eval_stage2_q3_v1.py; need=45000 ;;
    esac
    [ "$order" = swap ] && { extra="--swap_options"; tag="_swap"; }
    local out="outputs/${pfx}_${attr}_${cond}${tag}_v1.jsonl"
    if [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ]; then
      echo "[skip] $(basename $out)" | tee -a "$LOG"; continue
    fi
    while :; do
      local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu")
      [ "$free" -ge "$need" ] && break
      sleep 45
    done
    echo "[gpu$gpu] $(date +%H:%M) start $pfx $attr $cond $order" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=$gpu python3 "$script" --attr "$attr" --cond "$cond" $extra \
      >> "outputs/w15_${pfx}_${attr}_${cond}${tag}.out" 2>&1 || true
    echo "[gpu$gpu] $(date +%H:%M) done  $pfx $attr $cond $order rows=$( [ -f "$out" ] && wc -l < "$out" || echo 0)" | tee -a "$LOG"
  done
  echo "[gpu$gpu] empty" | tee -a "$LOG"
}
for g in 0 1; do worker "$g" & done
wait
echo "[wave15] all done $(date)" | tee -a "$LOG"
