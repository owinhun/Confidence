#!/bin/bash
# Screen the twelve graded degradations for the two operating points we need:
#   (a) the property is judgeable          -> unaided accuracy on the PROPERTY question
#   (b) the degradation impairs the task   -> unaided accuracy on the SOUND-EVENT question
# One 'none' run per set answers (a); the aux manifests answer (b) and are added once a
# candidate level is identified, so we do not spend GPU on levels that fail (a) outright.
set -u
cd /workspace/svhalluc/claude_track2
S=outputs/graded_status.txt
Q=/root/.claude/jobs/65f6dcfb/tmp/graded_queue
LK=/root/.claude/jobs/65f6dcfb/tmp/graded.lock
K(){ python3 -c "import datetime;print((datetime.datetime.utcnow()+datetime.timedelta(hours=9)).strftime('%H:%M'))"; }
log(){ echo "[KST $(K)] $*" >> "$S"; }
while pgrep -f "eval_stage2_q3_v1.py --attr sound --cond n_065" > /dev/null; do sleep 60; done
while pgrep -f "build_graded_attrs" > /dev/null; do sleep 30; done
log "prerequisites clear; graded screen starting"
: > "$Q"
for a in noise lowpass reverb; do for g in g1 g2 g3 g4; do
  echo "scripts/eval_stage2_factorial_v1.py|s2|27000|grad_${a}_${g}" >> "$Q"
  echo "scripts/eval_newmodels_v1.py|nm_qwen2audio|27000|grad_${a}_${g}" >> "$Q"
done; done
log "queued $(wc -l < "$Q")"
pop(){ exec 9>"$LK"; flock 9; local l; l=$(head -1 "$Q"); [ -n "$l" ] && sed -i '1d' "$Q"; flock -u 9; exec 9>&-; echo "$l"; }
worker(){ local wid=$1 g=$2
  while :; do
    local job; job=$(pop); [ -z "$job" ] && break
    IFS='|' read -r script prefix need attr <<< "$job"
    local out="outputs/${prefix}_${attr}_none_v1.jsonl" n=0
    for t in 1 2 3; do
      [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ] && { n=400; break; }
      while :; do f=$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits); [ "$f" -gt 28000 ] && break; sleep 60; done
      if [ "$prefix" = "nm_qwen2audio" ]; then
        CUDA_VISIBLE_DEVICES=$g python3 "$script" --model qwen2audio --attr "$attr" --cond none > "outputs/${prefix}_${attr}_none.log" 2>&1
      else
        CUDA_VISIBLE_DEVICES=$g python3 "$script" --attr "$attr" --cond none > "outputs/${prefix}_${attr}_none.log" 2>&1
      fi
      n=$(wc -l < "$out" 2>/dev/null||echo 0); [ "$n" -ge 400 ] && break; sleep 20
    done
    log "w$wid ${prefix}/${attr} -> $n"
  done; log "worker $wid done"; }
worker 1 0 & worker 2 1 & worker 3 2 & worker 4 3 &
wait
log "GRADED SCREEN DONE"
