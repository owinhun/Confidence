#!/bin/bash
# Robustness battery. Four manipulations, all on the sound-event cell where every model is
# evaluable, plus the two Qwen cells that were eligible on other attributes.
#   swap      (A) No (B) Yes -- an option-position bias cannot be mistaken for a result
#   hedge     directionless claim -- does destabilisation need the claim to point anywhere?
#   field     Reliability:/Certainty: instead of Confidence: -- is it the word or the content?
#   post      metadata after the question -- is it a recency effect?
set -u
cd /workspace/svhalluc/claude_track2
S=outputs/robust_status.txt
Q=/root/.claude/jobs/65f6dcfb/tmp/robust_queue
LK=/root/.claude/jobs/65f6dcfb/tmp/robust.lock
log(){ echo "[$(TZ=Asia/Seoul date +%H:%M)] $*" >> "$S"; }

: > "$Q"
# model|script|prefix|need
OMNI="qwen25|scripts/eval_stage2_factorial_v1.py|s2|27000
qwen3|scripts/eval_stage2_q3_v1.py|s2q3|78000
vllama|scripts/eval_stage2_vllama_v1.py|s2vl|27000"
NEWM="qwen2audio|scripts/eval_newmodels_v1.py|nm_qwen2audio|27000"
while IFS= read -r m; do
  for c in none anon_high anon_low; do echo "$m|$c|--swap_options|_swap" >> "$Q"; done
  echo "$m|hedge_plain||" >> "$Q"
  for c in reliab_high reliab_low certain_high certain_low; do echo "$m|$c||" >> "$Q"; done
  for c in anon_high anon_low none; do echo "$m|$c|--post|_post" >> "$Q"; done
done <<< "$OMNI"
while IFS= read -r m; do
  for c in none anon_high anon_low; do echo "$m|$c|--swap_options|_swap" >> "$Q"; done
  echo "$m|hedge_plain||" >> "$Q"
done <<< "$NEWM"
log "queued $(wc -l < "$Q") jobs"

pop(){ exec 9>"$LK"; flock 9; local l; l=$(head -1 "$Q"); [ -n "$l" ] && sed -i '1d' "$Q"; flock -u 9; exec 9>&-; echo "$l"; }
wait_free(){ local need=$1; shift; local g tot
  while :; do tot=0
    for g in "$@"; do tot=$((tot+$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits))); done
    [ "$tot" -gt "$need" ] && return 0; sleep 60
  done; }
worker(){ local wid=$1; shift; local gpus="$*" cvd
  cvd=$(echo "$gpus" | tr ' ' ,)
  while :; do
    local job; job=$(pop); [ -z "$job" ] && break
    IFS='|' read -r model script prefix need cond extra ftag <<< "$job"
    # the record is model|script|prefix|need|cond|extra|ftag
    local out="outputs/${prefix}_sound_${cond}${ftag}_v1.jsonl" n=0
    if [ "$need" -gt 40000 ] && [ "$(echo "$gpus" | wc -w)" -lt 2 ]; then
      exec 9>"$LK"; flock 9; echo "$job" >> "$Q"; flock -u 9; exec 9>&-; sleep 45; continue
    fi
    for t in 1 2 3; do
      [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ] && { n=400; break; }
      wait_free "$need" $gpus
      if [ "$prefix" = "nm_qwen2audio" ]; then
        CUDA_VISIBLE_DEVICES=$cvd python3 "$script" --model qwen2audio --attr sound --cond "$cond" $extra \
          > "outputs/${prefix}_sound_${cond}${ftag}.log" 2>&1
      else
        CUDA_VISIBLE_DEVICES=$cvd python3 "$script" --attr sound --cond "$cond" $extra \
          > "outputs/${prefix}_sound_${cond}${ftag}.log" 2>&1
      fi
      n=$(wc -l < "$out" 2>/dev/null||echo 0); [ "$n" -ge 400 ] && break; sleep 20
    done
    log "w$wid ${prefix}/${cond}${ftag} -> $n"
  done; log "worker $wid done"; }
worker 1 0 & worker 2 1 & worker 3 2 3 &
wait
log "ROBUSTNESS ALL DONE"
