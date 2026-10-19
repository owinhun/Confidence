#!/bin/bash
# Paraphrase test, re-run. The first attempt silently used the original question because the
# wiring patch never matched; the outputs were byte-identical to the non-paraphrase runs, which
# is how it was caught. Verified before launching this time.
set -u
cd /workspace/svhalluc/claude_track2
S=outputs/para_status.txt
Q=/root/.claude/jobs/65f6dcfb/tmp/para_queue
LK=/root/.claude/jobs/65f6dcfb/tmp/para.lock
K(){ python3 -c "import datetime;print((datetime.datetime.utcnow()+datetime.timedelta(hours=9)).strftime('%H:%M'))"; }
log(){ echo "[KST $(K)] $*" >> "$S"; }
: > "$Q"
for c in none anon_high anon_low; do
  echo "scripts/eval_stage2_factorial_v1.py|s2|27000|$c" >> "$Q"
  echo "scripts/eval_newmodels_v1.py|nm_qwen2audio|27000|$c" >> "$Q"
  echo "scripts/eval_stage2_q3_v1.py|s2q3|78000|$c" >> "$Q"
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
    local out="outputs/${prefix}_sound_${cond}_para_v1.jsonl" n=0
    for t in 1 2 3; do
      [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ] && { n=400; break; }
      wait_free "$need" $gpus
      if [ "$prefix" = "nm_qwen2audio" ]; then
        CUDA_VISIBLE_DEVICES=$cvd python3 "$script" --model qwen2audio --attr sound --cond "$cond" --paraphrase > "outputs/${prefix}_sound_${cond}_para.log" 2>&1
      else
        CUDA_VISIBLE_DEVICES=$cvd python3 "$script" --attr sound --cond "$cond" --paraphrase > "outputs/${prefix}_sound_${cond}_para.log" 2>&1
      fi
      n=$(wc -l < "$out" 2>/dev/null||echo 0); [ "$n" -ge 400 ] && break; sleep 20
    done
    log "w$wid ${prefix}/${cond}_para -> $n"
  done; log "worker $wid done"; }
worker 1 0 & worker 2 1 & worker 3 2 3 &
wait
log "PARAPHRASE RERUN DONE"
