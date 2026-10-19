#!/usr/bin/env bash
# Give the full matched-field battery to every admitted cell, so there is no selection left to
# explain. The reviews' strongest structural objection is that ten cells clear the eligibility
# gate but only seven carry the battery, with no stated principle for the split -- which leaves
# the central 2.1-3.8x result open to a cherry-picking reading. Running the missing conditions
# removes the choice rather than justifying it: field-name contrast goes from seven cells to ten.
#
# The three cells and what each lacks were computed from the outputs, not assumed:
#   s2/colour            priority_{high,low,true_high,true_low} x both orders
#   s2/grad_noise_g4     priority_true_{high,low} x both orders (plain priority already run)
#   s2q3/colour          priority_{high,low,true_high,true_low} x both orders
#
# GPUs 0/1 only -- 2/3 stay handed back to the other users on this machine.
set -uo pipefail
cd /workspace/svhalluc/claude_track2
Q=/root/.claude/jobs/65f6dcfb/tmp/w14.txt
LOCK=/root/.claude/jobs/65f6dcfb/tmp/w14.lock
LOG=outputs/wave14_log.txt
cp /root/.claude/jobs/65f6dcfb/tmp/w14_jobs.txt "$Q"
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
      >> "outputs/w14_${pfx}_${attr}_${cond}${tag}.out" 2>&1 || true
    echo "[gpu$gpu] $(date +%H:%M) done  $pfx $attr $cond $order rows=$( [ -f "$out" ] && wc -l < "$out" || echo 0)" | tee -a "$LOG"
  done
  echo "[gpu$gpu] empty" | tee -a "$LOG"
}
for g in 0 1; do worker "$g" & done
wait
echo "[wave14] all done $(date)" | tee -a "$LOG"
