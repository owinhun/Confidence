#!/usr/bin/env bash
# Light-job booster for wave20. The base runner is one job per GPU; the light models
# (7B: s2/fact, qwen2audio/new, ~30GB) can ride alongside a 45GB q3 job on an 81GB card.
# Light jobs are MOVED (under the same lock the base runner's take() uses) into a second
# queue, so every job lives in exactly one queue -- the duplicate-writer incident from an
# earlier wave came from a job visible to two workers, and this prevents the recurrence.
set -uo pipefail
cd /workspace/svhalluc/claude_track2
Q=/root/.claude/jobs/65f6dcfb/tmp/w20.txt
Q2=/root/.claude/jobs/65f6dcfb/tmp/w20light.txt
LOCK=/root/.claude/jobs/65f6dcfb/tmp/w20.lock
L2=/root/.claude/jobs/65f6dcfb/tmp/w20light.lock
LOG=outputs/wave20_log.txt

flock "$LOCK" bash -c "grep -E '^(s2|nm_qwen2audio) (fact|new) ' $Q > $Q2 || true
  grep -vE '^(s2|nm_qwen2audio) (fact|new) ' $Q > $Q.tmp && mv $Q.tmp $Q"
echo "[boost] light queue $(wc -l < "$Q2"), heavy left $(wc -l < "$Q")" | tee -a "$LOG"

take2() { flock "$L2" bash -c "l=\$(head -1 $Q2); [ -n \"\$l\" ] && sed -i 1d $Q2; echo \"\$l\""; }

lightworker() {
  local gpu=$1; sleep $((gpu*7))
  while :; do
    local job; job=$(take2); [ -z "$job" ] && break
    set -- $job
    local pfx=$1 kind=$2 cond=$3 ord=$4 script extra="" tag=""
    case $kind in
      fact) script=scripts/eval_stage2_factorial_v1.py ;;
      new)  script=scripts/eval_newmodels_v1.py; extra="--model qwen2audio" ;;
    esac
    [ "$ord" = swap ] && { extra="$extra --swap_options"; tag="_swap"; }
    local out="outputs/${pfx}_sound_${cond}${tag}_v1.jsonl"
    if [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ]; then
      echo "[skip] $(basename $out)" | tee -a "$LOG"; continue
    fi
    while :; do
      local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu")
      if [ "$free" -ge 32000 ]; then sleep 5
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu")
        [ "$free" -ge 32000 ] && break
      fi
      sleep 45
    done
    echo "[Bgpu$gpu] $(date +%H:%M) start $pfx $cond $ord" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=$gpu python3 "$script" --attr sound --cond "$cond" $extra \
      >> "outputs/w20_${pfx}_${cond}${tag}.out" 2>&1 || true
    echo "[Bgpu$gpu] $(date +%H:%M) done  $pfx $cond $ord rows=$( [ -f "$out" ] && wc -l < "$out" || echo 0)" | tee -a "$LOG"
  done
  echo "[Bgpu$gpu] light empty" | tee -a "$LOG"
}
for g in 0 1 2 3; do lightworker "$g" & done
wait
echo "[boost] light all done $(date)" | tee -a "$LOG"
