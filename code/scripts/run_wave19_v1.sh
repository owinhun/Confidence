#!/usr/bin/env bash
# Re-run of the four wave18 jobs that died at argument parsing.
#
# cascade3x_plain reached evidence() in all three scripts but only newmodels had it in the CONDS
# list argparse validates against, so factorial and q3 exited before loading a model and the
# runner's `|| true` logged them as done with zero rows. Same shape as the --swap_options failure
# earlier: the function was ready, the interface was not. This time both the accept and the
# reject were checked with a real invocation before queuing anything.
set -uo pipefail
cd /workspace/svhalluc/claude_track2
Q=/root/.claude/jobs/65f6dcfb/tmp/w19.txt
LOCK=/root/.claude/jobs/65f6dcfb/tmp/w19.lock
LOG=outputs/wave19_log.txt
: > "$Q"
for spec in "s2 fact" "s2q3 q3"; do
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
        >> "outputs/w19_${pfx}_${cond}${tag}.out" 2>&1 || true
    else
      CUDA_VISIBLE_DEVICES=$gpu python3 "$script" --attr sound --cond "$cond" $extra \
        >> "outputs/w19_${pfx}_${cond}${tag}.out" 2>&1 || true
    fi
    echo "[gpu$gpu] $(date +%H:%M) done  $pfx $cond $order rows=$( [ -f "$out" ] && wc -l < "$out" || echo 0)" | tee -a "$LOG"
  done
  echo "[gpu$gpu] empty" | tee -a "$LOG"
}
for g in 0 1 2 3; do worker "$g" & done
wait
echo "[wave19] all done $(date)" | tee -a "$LOG"
