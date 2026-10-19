#!/bin/bash
# TRUE-claim dose-response: the missing half of the utility curve.
#
# Everything measured so far is the harm a WRONG upstream detection does. A cascade
# designer cannot act on that alone, because at run time the claim's truth is unknown --
# only the detector's precision pi is. Expected accuracy is
#     Acc(c, pi) = pi * Acc_true(c) + (1 - pi) * Acc_false(c)
# and Acc_false(c) is already measured at five verbal and five numeric levels. This run
# supplies Acc_true(c) on the same items, which is what makes the encoding choice an
# optimisation rather than an observation.
set -u
CT2=/workspace/svhalluc/claude_track2; cd "$CT2"
S=/root/.claude/jobs/65f6dcfb/tmp/util_status.txt
Q=/root/.claude/jobs/65f6dcfb/tmp/util_queue
LOCK=/root/.claude/jobs/65f6dcfb/tmp/util_queue.lock
log(){ echo "[$(date -u +%H:%M)] $*" >> "$S"; }
NEED_FOR(){ case "$1" in qwen3) echo 78000;; *) echo 27000;; esac; }

: > "$Q"
for m in qwen25 vllama qwen3; do
  for c in v_vlow v_low v_mod v_high v_vhigh; do echo "$m $c" >> "$Q"; done
done

pop(){ local line; exec 9>"$LOCK"; flock 9
  line=$(head -n 1 "$Q"); [ -n "$line" ] && sed -i '1d' "$Q"
  flock -u 9; exec 9>&-; echo "$line"; }

wait_mem(){ local need=$1; shift; local g tot
  while :; do tot=0
    for g in "$@"; do
      tot=$((tot + $(nvidia-smi -i "$g" --query-gpu=memory.free --format=csv,noheader,nounits)))
    done
    [ "$tot" -gt "$need" ] && return 0; sleep 60
  done; }

worker(){ local wid=$1; shift; local gpus="$*" cvd
  cvd=$(echo "$gpus" | tr ' ' ',')
  while :; do
    local job; job=$(pop); [ -z "$job" ] && break
    set -- $job; local m=$1 c=$2
    # the 30B model cannot fit on one shared card; skip it in a single-card worker
    if [ "$m" = qwen3 ] && [ "$(echo "$gpus" | wc -w)" -lt 2 ]; then
      exec 9>"$LOCK"; flock 9; echo "$m $c" >> "$Q"; flock -u 9; exec 9>&-
      sleep 30; continue
    fi
    local out="outputs/cs_${m}_sound_${c}_T_v1.jsonl" n=0
    for try in 1 2 3; do
      if [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ]; then n=400; break; fi
      wait_mem "$(NEED_FOR "$m")" $gpus
      CUDA_VISIBLE_DEVICES=$cvd python3 scripts/eval_confscale_v1.py --model "$m" \
        --attr sound --cond "$c" --claim true > "outputs/cs_${m}_sound_${c}_T.log" 2>&1
      n=$(wc -l < "$out" 2>/dev/null || echo 0); [ "$n" -ge 400 ] && break; sleep 20
    done
    log "w$wid $m/$c/TRUE -> $n"
  done; log "worker $wid done"; }

worker 1 0 1 & worker 2 2 3 &
wait
log "UTILITY CURVES ALL DONE"
