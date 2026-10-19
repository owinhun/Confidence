#!/usr/bin/env bash
# The 7B half of wave8, started while Nemotron still holds GPU0 finishing wave7.
#
# Three cards were sitting idle behind a single serial Nemotron worker. These jobs need 30-45 GB
# and never touch GPU0, so they cannot displace it. The Nemotron half of wave8 is left for a
# separate runner that waits for wave7 to clear.
set -uo pipefail
cd /workspace/svhalluc/claude_track2
Q=/root/.claude/jobs/65f6dcfb/tmp/w8s.txt
LOCK=/root/.claude/jobs/65f6dcfb/tmp/w8s.lock
LOG=outputs/wave8_log.txt
: > "$Q"

# matched numeric rungs for the two 7B models that lack them
for spec in "qwen2audio new n_015 n_060 n_085" "af3 af3 n_015 n_060 n_085"; do
  set -- $spec; m=$1; k=$2; shift 2
  for c in "$@"; do
    for cl in false true; do echo "$m $k $c $cl" >> "$Q"; done
  done
done
# unstructured claim pair and the repaired empty-block rung
for spec in "qwen2audio new" "af3 af3" "s2 fact" "s2q3 q3" "s2vl vl"; do
  set -- $spec
  echo "$1 $2 bare_true none" >> "$Q"
  echo "$1 $2 null_priority none" >> "$Q"
done
echo "[queue-small] $(wc -l < "$Q") jobs" | tee -a "$LOG"

take() { flock "$LOCK" bash -c "l=\$(head -1 $Q); [ -n \"\$l\" ] && sed -i 1d $Q; echo \"\$l\""; }

worker() {
  local gpu=$1
  while :; do
    local job; job=$(take); [ -z "$job" ] && break
    set -- $job; local m=$1 k=$2 cond=$3 cl=$4
    local pfx="nm_$m" need=30000 script extra=""
    case $k in
      new)  script=scripts/eval_newmodels_v1.py; extra="--model $m" ;;
      af3)  script=scripts/run_af3_v1.sh ;;
      fact) script=scripts/eval_stage2_factorial_v1.py; pfx=$m ;;
      q3)   script=scripts/eval_stage2_q3_v1.py; pfx=$m; need=45000 ;;
      vl)   script=scripts/eval_stage2_vllama_v1.py; pfx=$m ;;
    esac
    local tag=""; [ "$cl" = true ] && tag="_T"
    local out="outputs/${pfx}_sound_${cond}${tag}_v1.jsonl"
    if [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ]; then
      echo "[skip] $(basename $out)" | tee -a "$LOG"; continue
    fi
    # bare_true and null_priority are distinct conditions, not claim-flag variants
    local cf="--claim $cl"; case $cond in bare_true|null_priority) cf="" ;; esac
    while :; do
      local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu")
      [ "$free" -ge "$need" ] && break
      sleep 45
    done
    echo "[gpu$gpu] $(date +%H:%M) start $pfx $cond claim=$cl" | tee -a "$LOG"
    if [[ $script == *.sh ]]; then
      CUDA_VISIBLE_DEVICES=$gpu bash "$script" --attr sound --cond "$cond" $cf \
        >> "outputs/w8_${pfx}_${cond}${tag}.out" 2>&1 || true
    else
      CUDA_VISIBLE_DEVICES=$gpu python3 "$script" --attr sound --cond "$cond" $cf $extra \
        >> "outputs/w8_${pfx}_${cond}${tag}.out" 2>&1 || true
    fi
    echo "[gpu$gpu] $(date +%H:%M) done  $pfx $cond claim=$cl rows=$( [ -f "$out" ] && wc -l < "$out" || echo 0)" | tee -a "$LOG"
  done
  echo "[gpu$gpu] small queue empty" | tee -a "$LOG"
}

for g in 1 2 3; do worker "$g" & done
wait
echo "[wave8-small] all done $(date)" | tee -a "$LOG"
