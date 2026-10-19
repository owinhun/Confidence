#!/bin/bash
# Phase 3, queued behind the robustness battery.
#   para       question reworded so it shares no content words with the claim -- separates
#              "proposition-specific" from "lexical-overlap-specific"
#   otherclip  the claim is about a DIFFERENT clip -- is it bound to the input at all?
#   swap       the option-order control extended to every remaining evaluable cell, after it
#              withdrew VideoLLaMA's sound cells
set -u
cd /workspace/svhalluc/claude_track2
S=outputs/phase3_status.txt
Q=/root/.claude/jobs/65f6dcfb/tmp/phase3_queue
LK=/root/.claude/jobs/65f6dcfb/tmp/phase3.lock
K(){ python3 -c "import datetime;print((datetime.datetime.utcnow()+datetime.timedelta(hours=9)).strftime('%H:%M'))"; }
log(){ echo "[KST $(K)] $*" >> "$S"; }

# wait for the robustness battery to drain
while [ "$(pgrep -cf 'eval_stage2|eval_newmodels' 2>/dev/null || echo 0)" -gt 0 ]; do sleep 120; done
log "robustness drained; phase 3 starting"

: > "$Q"
# script|prefix|need|attr|cond|extra|ftag
for m in "scripts/eval_stage2_factorial_v1.py|s2|27000" \
         "scripts/eval_stage2_q3_v1.py|s2q3|78000" \
         "scripts/eval_newmodels_v1.py|nm_qwen2audio|27000"; do
  for c in none anon_high anon_low; do echo "$m|sound|$c|--paraphrase|_para" >> "$Q"; done
  for c in otherclip_high otherclip_low; do echo "$m|sound|$c||" >> "$Q"; done
done
# extend the option-order control to the remaining evaluable cells
echo "scripts/eval_stage2_q3_v1.py|s2q3|78000|noise|none|--swap_options|_swap" >> "$Q"
echo "scripts/eval_stage2_q3_v1.py|s2q3|78000|noise|anon_high|--swap_options|_swap" >> "$Q"
echo "scripts/eval_stage2_q3_v1.py|s2q3|78000|noise|anon_low|--swap_options|_swap" >> "$Q"
for c in none anon_high anon_low; do
  echo "scripts/eval_stage2_factorial_v1.py|s2|27000|colour|$c|--swap_options|_swap" >> "$Q"
  echo "scripts/eval_stage2_q3_v1.py|s2q3|78000|colour|$c|--swap_options|_swap" >> "$Q"
done
# VideoLLaMA's greyscale/expert contrast is one of the two largest in the paper but only its
# anonymous cell has a swapped read-out so far; without these it drops out of the v2 table.
for c in expert_high expert_low; do
  echo "scripts/eval_stage2_vllama_v1.py|s2vl|27000|colour|$c|--swap_options|_swap" >> "$Q"
done
log "queued $(wc -l < "$Q") jobs"

pop(){ exec 9>"$LK"; flock 9; local l; l=$(head -1 "$Q"); [ -n "$l" ] && sed -i '1d' "$Q"; flock -u 9; exec 9>&-; echo "$l"; }
wait_free(){ local need=$1; shift; local g tot
  while :; do tot=0
    for g in "$@"; do tot=$((tot+$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits))); done
    [ "$tot" -gt "$need" ] && return 0; sleep 60
  done; }
worker(){ local wid=$1; shift; local gpus="$*" cvd; cvd=$(echo "$gpus"|tr ' ' ,)
  while :; do
    local job; job=$(pop); [ -z "$job" ] && break
    IFS='|' read -r script prefix need attr cond extra ftag <<< "$job"
    if [ "$need" -gt 40000 ] && [ "$(echo "$gpus"|wc -w)" -lt 2 ]; then
      exec 9>"$LK"; flock 9; echo "$job" >> "$Q"; flock -u 9; exec 9>&-; sleep 45; continue
    fi
    local out="outputs/${prefix}_${attr}_${cond}${ftag}_v1.jsonl" n=0
    for t in 1 2 3; do
      [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ] && { n=400; break; }
      wait_free "$need" $gpus
      if [ "$prefix" = "nm_qwen2audio" ]; then
        CUDA_VISIBLE_DEVICES=$cvd python3 "$script" --model qwen2audio --attr "$attr" --cond "$cond" $extra > "outputs/${prefix}_${attr}_${cond}${ftag}.log" 2>&1
      else
        CUDA_VISIBLE_DEVICES=$cvd python3 "$script" --attr "$attr" --cond "$cond" $extra > "outputs/${prefix}_${attr}_${cond}${ftag}.log" 2>&1
      fi
      n=$(wc -l < "$out" 2>/dev/null||echo 0); [ "$n" -ge 400 ] && break; sleep 20
    done
    log "w$wid ${prefix}/${attr}/${cond}${ftag} -> $n"
  done; log "worker $wid done"; }
worker 1 0 & worker 2 1 & worker 3 2 3 &
wait
log "PHASE 3 ALL DONE"
