#!/bin/bash
# Controls demanded by both reviews and not previously run:
#   null_block   an empty structured block -- isolates regression-to-the-mean / any-perturbation
#   priority_*   a field with no epistemic content -- separates the word from the meaning
#   n_055..n_075 numeric levels all above 0.5, so "low confidence asserts the complement"
#                cannot explain the effect
set -u
cd /workspace/svhalluc/claude_track2
S=outputs/revctl_status.txt
Q=/root/.claude/jobs/65f6dcfb/tmp/revctl_queue
LK=/root/.claude/jobs/65f6dcfb/tmp/revctl.lock
K(){ python3 -c "import datetime;print((datetime.datetime.utcnow()+datetime.timedelta(hours=9)).strftime('%H:%M'))"; }
log(){ echo "[KST $(K)] $*" >> "$S"; }
# wait for the gap-fill batch to drain
while [ "$(pgrep -cf 'eval_newmodels|eval_af3' 2>/dev/null || echo 0)" -gt 0 ]; do sleep 120; done
log "gap-fill drained; review controls starting"
: > "$Q"
for spec in "scripts/eval_stage2_factorial_v1.py|s2|27000" \
            "scripts/eval_stage2_q3_v1.py|s2q3|78000" \
            "scripts/eval_newmodels_v1.py|nm_qwen2audio|27000"; do
  for c in null_block priority_high priority_low n_055 n_065 n_075; do
    echo "$spec|$c" >> "$Q"
  done
done
log "queued $(wc -l < "$Q")"
pop(){ exec 9>"$LK"; flock 9; local l; l=$(head -1 "$Q"); [ -n "$l" ] && sed -i '1d' "$Q"; flock -u 9; exec 9>&-; echo "$l"; }
wait_free(){ local need=$1; shift; local g tot
  while :; do tot=0
    for g in "$@"; do tot=$((tot+$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits))); done
    [ "$tot" -gt "$need" ] && return 0; sleep 60
  done; }
worker(){ local wid=$1; shift; local gpus="$*" cvd; cvd=$(echo "$gpus"|tr ' ' ,)
  while :; do
    local job; job=$(pop); [ -z "$job" ] && break
    IFS='|' read -r script prefix need cond <<< "$job"
    if [ "$need" -gt 40000 ] && [ "$(echo "$gpus"|wc -w)" -lt 2 ]; then
      exec 9>"$LK"; flock 9; echo "$job" >> "$Q"; flock -u 9; exec 9>&-; sleep 45; continue
    fi
    local out="outputs/${prefix}_sound_${cond}_v1.jsonl" n=0
    for t in 1 2 3; do
      [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ] && { n=400; break; }
      wait_free "$need" $gpus
      if [ "$prefix" = "nm_qwen2audio" ]; then
        CUDA_VISIBLE_DEVICES=$cvd python3 "$script" --model qwen2audio --attr sound --cond "$cond" > "outputs/${prefix}_sound_${cond}.log" 2>&1
      else
        CUDA_VISIBLE_DEVICES=$cvd python3 "$script" --attr sound --cond "$cond" > "outputs/${prefix}_sound_${cond}.log" 2>&1
      fi
      n=$(wc -l < "$out" 2>/dev/null||echo 0); [ "$n" -ge 400 ] && break; sleep 20
    done
    log "w$wid ${prefix}/${cond} -> $n"
  done; log "worker $wid done"; }
worker 1 0 & worker 2 1 & worker 3 2 3 &
wait
log "REVIEW CONTROLS DONE"
