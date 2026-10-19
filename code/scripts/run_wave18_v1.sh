#!/usr/bin/env bash
# A detector from a second family, not a second checkpoint of the same one.
#
# Both cascade detectors so far are CLAP. That tests checkpoint sensitivity, not family
# sensitivity, which is what the reviews ask about. AST and PANNs cannot serve -- they answer in
# AudioSet's vocabulary while the manifest uses VGGSound's, and the mapping is a project of its
# own. Qwen2-Audio is asked the query class verbatim, so no mapping enters, and its unaided
# accuracy (77.0%, out-of-fold 77.8%) sits at a comparable operating point to CLAP's 74.8%.
#
# Its own cell is excluded downstream: a model reading back a claim it produced itself is not a
# cascade. That leaves the three other order-robust downstream cells.
set -uo pipefail
cd /workspace/svhalluc/claude_track2
Q=/root/.claude/jobs/65f6dcfb/tmp/w18.txt
LOCK=/root/.claude/jobs/65f6dcfb/tmp/w18.lock
LOG=outputs/wave18_log.txt
: > "$Q"
for spec in "s2 fact" "s2q3 q3" "nm_nemotron nemo"; do
  set -- $spec
  for o in none swap; do echo "$1 $2 cascade3x_plain $o" >> "$Q"; done
done
echo "[queue] $(wc -l < "$Q") jobs $(date)" | tee -a "$LOG"
take() { flock "$LOCK" bash -c "l=\$(head -1 $Q); [ -n \"\$l\" ] && sed -i 1d $Q; echo \"\$l\""; }
worker() {
  local gpu=$1
  while :; do
    local job; job=$(take); [ -z "$job" ] && break
    set -- $job; local pfx=$1 kind=$2 cond=$3 order=$4
    local script need=30000 extra="" tag=""
    case $kind in
      fact) script=scripts/eval_stage2_factorial_v1.py ;;
      q3)   script=scripts/eval_stage2_q3_v1.py; need=45000 ;;
      nemo) script=scripts/run_nemotron_v1.sh; need=65000 ;;
    esac
    [ "$order" = swap ] && { extra="--swap_options"; tag="_swap"; }
    local out="outputs/${pfx}_sound_${cond}${tag}_v1.jsonl"
    if [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ]; then
      echo "[skip] $(basename $out)" | tee -a "$LOG"; continue
    fi
    while :; do
      local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu")
      [ "$free" -ge "$need" ] && break
      sleep 45
    done
    echo "[gpu$gpu] $(date +%H:%M) start $pfx $cond $order" | tee -a "$LOG"
    if [[ $script == *.sh ]]; then
      CUDA_VISIBLE_DEVICES=$gpu bash "$script" --attr sound --cond "$cond" $extra \
        >> "outputs/w18_${pfx}_${cond}${tag}.out" 2>&1 || true
    else
      CUDA_VISIBLE_DEVICES=$gpu python3 "$script" --attr sound --cond "$cond" $extra \
        >> "outputs/w18_${pfx}_${cond}${tag}.out" 2>&1 || true
    fi
    echo "[gpu$gpu] $(date +%H:%M) done  $pfx $cond $order rows=$( [ -f "$out" ] && wc -l < "$out" || echo 0)" | tee -a "$LOG"
  done
  echo "[gpu$gpu] empty" | tee -a "$LOG"
}
for g in 0 1 2 3; do worker "$g" & done
wait
echo "[wave18] all done $(date)" | tee -a "$LOG"
