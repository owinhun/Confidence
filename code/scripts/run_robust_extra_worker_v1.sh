#!/bin/bash
# Second two-GPU worker on cards 0+1: the tail of the queue is all Qwen3-30B, which needs a
# pair, so the single-card workers can only push those jobs back and idle.
set -u
cd /workspace/svhalluc/claude_track2
S=outputs/robust_status.txt
Q=/root/.claude/jobs/65f6dcfb/tmp/robust_queue
LK=/root/.claude/jobs/65f6dcfb/tmp/robust.lock
log(){ echo "[$(TZ=Asia/Seoul date +%H:%M)] $*" >> "$S"; }
pop(){ exec 9>"$LK"; flock 9; local l; l=$(head -1 "$Q"); [ -n "$l" ] && sed -i '1d' "$Q"; flock -u 9; exec 9>&-; echo "$l"; }
while :; do
  job=$(pop); [ -z "$job" ] && break
  IFS='|' read -r model script prefix need cond extra ftag <<< "$job"
  out="outputs/${prefix}_sound_${cond}${ftag}_v1.jsonl"; n=0
  for t in 1 2 3; do
    [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ] && { n=400; break; }
    while :; do
      f0=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits)
      f1=$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits)
      [ $((f0+f1)) -gt "$need" ] && break; sleep 60
    done
    CUDA_VISIBLE_DEVICES=0,1 python3 "$script" --attr sound --cond "$cond" $extra \
      > "outputs/${prefix}_sound_${cond}${ftag}.log" 2>&1
    n=$(wc -l < "$out" 2>/dev/null||echo 0); [ "$n" -ge 400 ] && break; sleep 20
  done
  log "w4(0,1) ${prefix}/${cond}${ftag} -> $n"
done
log "worker 4 done"
