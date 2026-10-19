#!/usr/bin/env bash
# Wave 4. Two batteries, one queue, workers gated on free memory rather than on each other --
# a global barrier has left these GPUs idle three times now.
#
#   cascade_mono / cascade_pct   the repaired forward. Forwarding the detector's raw word cost
#                                every model 1.0-12.2 points, and the cascade table blames the
#                                bands being non-monotone in precision. These two say whether
#                                that is the whole story.
#   silence                      a purely auditory attribute the paper has never scored. Qwen2.5
#                                (A0 73.5) and Qwen3 (78.0) clear the gate unswapped; the swapped
#                                pass decides whether either cell survives order-averaging.
set -uo pipefail
cd /workspace/svhalluc/claude_track2
Q=/root/.claude/jobs/65f6dcfb/tmp/wave4_queue.txt
LOCK=/root/.claude/jobs/65f6dcfb/tmp/wave4.lock
LOG=outputs/wave4_log.txt
: > "$Q"

# repaired cascade, on the five cells the cascade table already reports
for m in "s2 sound factorial" "s2q3 sound q3" "nm_qwen2audio sound new" \
         "nm_af3 sound af3" "nm_nemotron sound nemotron"; do
  set -- $m
  for c in cascade_mono cascade_pct; do echo "$1 $2 $3 $c none" >> "$Q"; done
done
# silence: the swapped baseline first, then the battery on both orders
for m in "s2 factorial" "s2q3 q3"; do
  set -- $m
  echo "$1 silence $2 none swap" >> "$Q"
  for c in anon_high anon_low true_high true_low priority_high priority_low \
           priority_true_high priority_true_low; do
    echo "$1 silence $2 $c none" >> "$Q"
    echo "$1 silence $2 $c swap" >> "$Q"
  done
done
echo "[queue] $(wc -l < "$Q") jobs" | tee -a "$LOG"

take() { flock "$LOCK" bash -c 'f='"$Q"'; l=$(head -1 "$f"); [ -n "$l" ] && sed -i 1d "$f"; echo "$l"'; }

worker() {
  local gpu=$1
  while :; do
    local job; job=$(take); [ -z "$job" ] && break
    set -- $job; local pfx=$1 attr=$2 kind=$3 cond=$4 order=$5
    local need=30000
    [ "$pfx" = nm_nemotron ] && need=65000
    while :; do
      local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu")
      [ "$free" -ge "$need" ] && break
      sleep 60
    done
    local tag=""; [ "$order" = swap ] && tag="_swap"
    local out="outputs/${pfx}_${attr}_${cond}${tag}_v1.jsonl"
    # a file another worker is actively appending to must not be picked up as done
    if [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ]; then
      echo "[skip] $out" | tee -a "$LOG"; continue
    fi
    local script; case $kind in
      factorial) script=scripts/eval_stage2_factorial_v1.py ;;
      q3)        script=scripts/eval_stage2_q3_v1.py ;;
      new)       script=scripts/eval_newmodels_v1.py ;;
      af3)       script=scripts/eval_af3_v1.py ;;
      nemotron)  script=scripts/eval_nemotron_v1.py ;;
    esac
    local extra=""; [ "$order" = swap ] && extra="--swap_options"
    # eval_newmodels dispatches on --model and prefixes the output with nm_; the AF3 and
    # Nemotron ports have their own entry points and take no such flag
    [ "$kind" = new ] && extra="$extra --model ${pfx#nm_}"
    echo "[gpu$gpu] $(date +%H:%M) start $pfx $attr $cond $order" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=$gpu python3 "$script" --attr "$attr" --cond "$cond" $extra \
      >> "outputs/wave4_${pfx}_${attr}_${cond}${tag}.out" 2>&1 || true
    echo "[gpu$gpu] $(date +%H:%M) done  $pfx $attr $cond $order rows=$( [ -f "$out" ] && wc -l < "$out" || echo 0)" | tee -a "$LOG"
  done
  echo "[gpu$gpu] queue empty" | tee -a "$LOG"
}

for g in 0 1 2 3; do worker "$g" & done
wait
echo "[wave4] all done $(date)" | tee -a "$LOG"
