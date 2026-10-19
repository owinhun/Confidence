#!/bin/bash
# Two more workers on the graded queue. They pop through the same flock as the original four,
# so a job is still handed to exactly one worker; the only thing that changes is how many
# cards are being fed. GPUs 0 and 1 have ~55 GB free against a ~25 GB job, so each can carry a
# second concurrent run. The >28 GB guard is unchanged, so if a card fills these simply wait.
set -u
cd /workspace/svhalluc/claude_track2
S=outputs/graded_status.txt
Q=/root/.claude/jobs/65f6dcfb/tmp/graded_queue
LK=/root/.claude/jobs/65f6dcfb/tmp/graded.lock
K(){ python3 -c "import datetime;print((datetime.datetime.utcnow()+datetime.timedelta(hours=9)).strftime('%H:%M'))"; }
log(){ echo "[KST $(K)] $*" >> "$S"; }
pop(){ exec 9>"$LK"; flock 9; local l; l=$(head -1 "$Q"); [ -n "$l" ] && sed -i '1d' "$Q"; flock -u 9; exec 9>&-; echo "$l"; }
worker(){ local wid=$1 g=$2
  while :; do
    local job; job=$(pop); [ -z "$job" ] && break
    IFS='|' read -r script prefix need attr <<< "$job"
    local out="outputs/${prefix}_${attr}_none_v1.jsonl" n=0
    for t in 1 2 3; do
      [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ] && { n=400; break; }
      while :; do f=$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits); [ "$f" -gt 28000 ] && break; sleep 45; done
      if [ "$prefix" = "nm_qwen2audio" ]; then
        CUDA_VISIBLE_DEVICES=$g python3 "$script" --model qwen2audio --attr "$attr" --cond none > "outputs/${prefix}_${attr}_none.log" 2>&1
      else
        CUDA_VISIBLE_DEVICES=$g python3 "$script" --attr "$attr" --cond none > "outputs/${prefix}_${attr}_none.log" 2>&1
      fi
      n=$(wc -l < "$out" 2>/dev/null||echo 0); [ "$n" -ge 400 ] && break; sleep 20
    done
    log "x$wid ${prefix}/${attr} -> $n"
  done; log "extra worker $wid done"; }
worker 7 0 & worker 8 1 &
wait
