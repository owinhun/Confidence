#!/usr/bin/env bash
# One scheduler for all remaining work.
#
# Four independent runners were competing for the same four cards with different memory needs,
# and the 30B Nemotron -- which wants a nearly empty card -- was starved behind a queue of 7B
# jobs that kept refilling. Its ETA had drifted four hours past everything else. This runner
# owns every job, knows each model's footprint and entry point, and gives Nemotron a dedicated
# worker so it starts immediately instead of waiting for a gap that never comes.
#
# Worker 0 runs only Nemotron (needs ~60 GB). Workers 1-3 take everything else.
set -uo pipefail
cd /workspace/svhalluc/claude_track2
QN=/root/.claude/jobs/65f6dcfb/tmp/w5_nemo.txt
QS=/root/.claude/jobs/65f6dcfb/tmp/w5_small.txt
LOCK=/root/.claude/jobs/65f6dcfb/tmp/w5.lock
LOG=outputs/wave5_log.txt
: > "$QN"; : > "$QS"

# Nemotron: corrected cascade and the claim-only branches, both orders
for o in none swap; do
  for c in cascade_bare cascade_conf cascade_pct detclaim_true detclaim_false; do
    echo "nemotron $c $o" >> "$QN"
  done
done
# AF3: same set, through its own wrapper
for o in none swap; do
  for c in cascade_bare cascade_conf cascade_pct detclaim_true detclaim_false; do
    echo "af3 $c $o" >> "$QS"
  done
done
# everything the held wave4 queue still wanted
awk '{tag=($5=="swap")?"_swap":""; print $1, $2, $3, $4, $5}' \
  /root/.claude/jobs/65f6dcfb/tmp/wave5_pending.txt >> "$QS"
echo "[queue] nemotron $(wc -l < "$QN")  other $(wc -l < "$QS")" | tee -a "$LOG"

take() { flock "$LOCK" bash -c "f=\$1; l=\$(head -1 \"\$f\"); [ -n \"\$l\" ] && sed -i 1d \"\$f\"; echo \"\$l\"" _ "$1"; }

wait_mem() {
  local gpu=$1 need=$2
  while :; do
    local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu")
    [ "$free" -ge "$need" ] && break
    sleep 45
  done
}

run_job() {
  local gpu=$1; shift
  local a=$1 b=$2 c=$3 d=${4:-} e=${5:-}
  local pfx attr cond order script extra need out
  if [ -z "$d" ]; then                 # "<model> <cond> <order>" form
    case $a in
      nemotron) pfx=nm_nemotron; script=scripts/run_nemotron_v1.sh; need=65000 ;;
      af3)      pfx=nm_af3;      script=scripts/run_af3_v1.sh;      need=30000 ;;
    esac
    attr=sound; cond=$b; order=$c; extra=""
  else                                  # "<pfx> <attr> <kind> <cond> <order>" form
    pfx=$a; attr=$b; cond=$d; order=$e; need=30000
    case $c in
      factorial) script=scripts/eval_stage2_factorial_v1.py ;;
      q3)        script=scripts/eval_stage2_q3_v1.py; need=45000 ;;
      new)       script=scripts/eval_newmodels_v1.py ;;
    esac
    extra="--model ${pfx#nm_}"; [ "$c" != new ] && extra=""
  fi
  local tag=""; [ "$order" = swap ] && tag="_swap"
  out="outputs/${pfx}_${attr}_${cond}${tag}_v1.jsonl"
  if [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ]; then
    echo "[skip] $(basename $out)" | tee -a "$LOG"; return
  fi
  [ "$order" = swap ] && extra="$extra --swap_options"
  wait_mem "$gpu" "$need"
  echo "[gpu$gpu] $(date +%H:%M) start $pfx $attr $cond $order" | tee -a "$LOG"
  if [[ $script == *.sh ]]; then
    CUDA_VISIBLE_DEVICES=$gpu bash "$script" --attr "$attr" --cond "$cond" $extra \
      >> "outputs/w5_${pfx}_${attr}_${cond}${tag}.out" 2>&1 || true
  else
    CUDA_VISIBLE_DEVICES=$gpu python3 "$script" --attr "$attr" --cond "$cond" $extra \
      >> "outputs/w5_${pfx}_${attr}_${cond}${tag}.out" 2>&1 || true
  fi
  echo "[gpu$gpu] $(date +%H:%M) done  $pfx $attr $cond $order rows=$( [ -f "$out" ] && wc -l < "$out" || echo 0)" | tee -a "$LOG"
}

worker() {
  local gpu=$1 q=$2
  while :; do
    local job; job=$(take "$q"); [ -z "$job" ] && break
    run_job "$gpu" $job
  done
  echo "[gpu$gpu] queue empty" | tee -a "$LOG"
}

worker 0 "$QN" &
for g in 1 2 3; do worker "$g" "$QS" & done
wait
echo "[wave5] all done $(date)" | tee -a "$LOG"
