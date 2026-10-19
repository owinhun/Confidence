#!/bin/bash
# Qwen3-30B needs a whole GPU pair; two workers racing for one produced repeated OOM during
# shard loading. Run its review controls strictly one at a time.
set -u
cd /workspace/svhalluc/claude_track2
S=outputs/revctl_status.txt
K(){ python3 -c "import datetime;print((datetime.datetime.utcnow()+datetime.timedelta(hours=9)).strftime('%H:%M'))"; }
log(){ echo "[KST $(K)] $*" >> "$S"; }
while pgrep -f "run_review_controls_v1.sh" > /dev/null; do sleep 60; done
pick(){ local g h f fh
  while :; do
    for g in 0 1 2 3; do for h in 0 1 2 3; do
      [ "$g" -ge "$h" ] && continue
      f=$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits)
      fh=$(nvidia-smi -i $h --query-gpu=memory.free --format=csv,noheader,nounits)
      [ $((f+fh)) -gt 84000 ] && { echo "$g,$h"; return 0; }
    done; done
    sleep 90
  done; }
for c in priority_low null_block priority_high n_065 n_075; do
  out="outputs/s2q3_sound_${c}_v1.jsonl"
  for t in 1 2 3; do
    [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ] && break
    P=$(pick)
    CUDA_VISIBLE_DEVICES=$P python3 scripts/eval_stage2_q3_v1.py --attr sound --cond "$c" \
      > "outputs/s2q3_sound_${c}.log" 2>&1
    n=$(wc -l < "$out" 2>/dev/null||echo 0); log "serial q3/$c try$t [$P] -> $n"
    [ "$n" -ge 400 ] && break; sleep 30
  done
done
log "Q3 REVIEW CONTROLS SERIAL DONE"
