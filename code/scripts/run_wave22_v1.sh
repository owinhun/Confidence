#!/usr/bin/env bash
# Tier-T: genuinely stronger detector, bare forwarding. Pre-registered in outputs/log_v1.md.
# Pre-registered in outputs/log_v1.md (2026-09-01 20:33 KST) before any job ran.
# 4 mix levels x 4 order-robust cells x 2 answer orders = 32 jobs, GPUs 0-3 only.
set -uo pipefail
cd /workspace/svhalluc/claude_track2
Q=/root/.claude/jobs/65f6dcfb/tmp/w22.txt
LOCK=/root/.claude/jobs/65f6dcfb/tmp/w22.lock
LOG=outputs/wave22_log.txt
: > "$Q"
for spec in "s2 fact" "s2q3 q3" "nm_qwen2audio new" "nm_nemotron nemo"; do
  set -- $spec
  for o in none swap; do echo "$1 $2 cascademixstr_plain $o" >> "$Q"; done
done
echo "[wave22] queue $(wc -l < "$Q") jobs $(date)" | tee -a "$LOG"

take() { flock "$LOCK" bash -c "l=\$(head -1 $Q); [ -n \"\$l\" ] && sed -i 1d $Q; echo \"\$l\""; }

worker() {
  local gpu=$1
  while :; do
    local job; job=$(take); [ -z "$job" ] && break
    set -- $job
    local pfx=$1 kind=$2 cond=$3 ord=$4
    local script need=30000 extra="" tag=""
    case $kind in
      fact) script=scripts/eval_stage2_factorial_v1.py ;;
      q3)   script=scripts/eval_stage2_q3_v1.py; need=45000 ;;
      new)  script=scripts/eval_newmodels_v1.py; extra="--model qwen2audio" ;;
      nemo) script=scripts/run_nemotron_v1.sh; need=65000 ;;
    esac
    [ "$ord" = swap ] && { extra="$extra --swap_options"; tag="_swap"; }
    local out="outputs/${pfx}_sound_${cond}${tag}_v1.jsonl"
    if [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ]; then
      echo "[skip] $(basename $out)" | tee -a "$LOG"; continue
    fi
    while :; do
      local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu")
      [ "$free" -ge "$need" ] && break
      sleep 45
    done
    echo "[gpu$gpu] $(date +%H:%M) start $pfx $cond $ord" | tee -a "$LOG"
    if [[ $script == *.sh ]]; then
      CUDA_VISIBLE_DEVICES=$gpu bash "$script" --attr sound --cond "$cond" $extra \
        >> "outputs/w20_${pfx}_${cond}${tag}.out" 2>&1 || true
    else
      CUDA_VISIBLE_DEVICES=$gpu python3 "$script" --attr sound --cond "$cond" $extra \
        >> "outputs/w20_${pfx}_${cond}${tag}.out" 2>&1 || true
    fi
    echo "[gpu$gpu] $(date +%H:%M) done  $pfx $cond $ord rows=$( [ -f "$out" ] && wc -l < "$out" || echo 0)" | tee -a "$LOG"
  done
  echo "[gpu$gpu] empty" | tee -a "$LOG"
}
for g in 0 1; do worker "$g" & done
wait
echo "[wave22] all done $(date)" | tee -a "$LOG"
