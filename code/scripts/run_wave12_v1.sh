#!/usr/bin/env bash
# Fill in the one cell that made Figure 1's seven and Table 1's seven different sets.
#
# Qwen3/noise clears the both-order gate and carries the override result, so it is a primary row
# in Table 1. It never reached Figure 1 because that figure needs the five-rung confidence ladder
# and this cell only ever had anon_high/anon_low -- two rungs. The two "sevens" therefore named
# different cells, which is the reviews' D-1.
#
# Documenting the mismatch in a caption would be honest but leaves it in place. Running the
# missing conditions removes it: the ladder puts the cell in Figure 1, and the matched
# non-epistemic controls make it a full battery cell, taking the field-name result from six
# order-robust cells to seven. The GPUs are idle, so this costs nothing but wall-clock.
set -uo pipefail
cd /workspace/svhalluc/claude_track2
Q=/root/.claude/jobs/65f6dcfb/tmp/w12.txt
LOCK=/root/.claude/jobs/65f6dcfb/tmp/w12.lock
LOG=outputs/wave12_log.txt
: > "$Q"

# 1. the five-rung ladder, both claim directions, both answer orders  (20)
for c in v_vlow v_low v_mod v_high v_vhigh; do
  for cl in false true; do
    for o in none swap; do echo "cs $c $cl $o" >> "$Q"; done
  done
done
# 2. the matched non-epistemic controls that make it a battery cell, both orders  (12)
for c in priority_high priority_low urgency_high urgency_low opaque_high opaque_low; do
  for o in none swap; do echo "q3 $c none $o" >> "$Q"; done
done
echo "[queue] $(wc -l < "$Q") jobs $(date)" | tee -a "$LOG"

take() { flock "$LOCK" bash -c "l=\$(head -1 $Q); [ -n \"\$l\" ] && sed -i 1d $Q; echo \"\$l\""; }

worker() {
  local gpu=$1
  while :; do
    local job; job=$(take); [ -z "$job" ] && break
    set -- $job
    local kind=$1 cond=$2 claim=$3 order=$4
    local script extra="" pfx tag="" out
    if [ "$kind" = cs ]; then
      script=scripts/eval_confscale_v1.py; pfx=cs_qwen3; extra="--model qwen3"
      [ "$claim" = true ] && { extra="$extra --claim true"; tag="_T"; }
    else
      script=scripts/eval_stage2_q3_v1.py; pfx=s2q3
    fi
    [ "$order" = swap ] && { extra="$extra --swap_options"; tag="${tag}_swap"; }
    out="outputs/${pfx}_noise_${cond}${tag}_v1.jsonl"
    if [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ]; then
      echo "[skip] $(basename $out)" | tee -a "$LOG"; continue
    fi
    while :; do
      local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu")
      [ "$free" -ge 45000 ] && break
      sleep 45
    done
    echo "[gpu$gpu] $(date +%H:%M) start $pfx noise $cond claim=$claim $order" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=$gpu python3 "$script" --attr noise --cond "$cond" $extra \
      >> "outputs/w12_${pfx}_${cond}${tag}.out" 2>&1 || true
    echo "[gpu$gpu] $(date +%H:%M) done  $pfx noise $cond claim=$claim $order rows=$( [ -f "$out" ] && wc -l < "$out" || echo 0)" | tee -a "$LOG"
  done
  echo "[gpu$gpu] empty" | tee -a "$LOG"
}

for g in 0 1 2 3; do worker "$g" & done
wait
echo "[wave12] all done $(date)" | tee -a "$LOG"
