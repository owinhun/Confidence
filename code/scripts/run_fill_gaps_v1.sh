#!/bin/bash
# Fill every cell the completeness audit found missing on the two audio-only models. These
# were skipped by ordering, not by design: the robustness battery and the withheld-signal
# control were built before Qwen2-Audio and Audio Flamingo 3 entered the study.
set -u
cd /workspace/svhalluc/claude_track2
S=outputs/fill_status.txt
Q=/root/.claude/jobs/65f6dcfb/tmp/fill_queue
LK=/root/.claude/jobs/65f6dcfb/tmp/fill.lock
K(){ python3 -c "import datetime;print((datetime.datetime.utcnow()+datetime.timedelta(hours=9)).strftime('%H:%M'))"; }
log(){ echo "[KST $(K)] $*" >> "$S"; }
: > "$Q"
# model|cond|extra|ftag
for m in qwen2audio af3; do
  echo "$m|anon_high|--mask_sensory|_masked" >> "$Q"
  for c in reliab_high reliab_low certain_high certain_low otherclip_high otherclip_low; do
    echo "$m|$c||" >> "$Q"
  done
  for c in none anon_high anon_low; do echo "$m|$c|--post|_post" >> "$Q"; done
done
# AF3 also lacks swap, paraphrase and the hedge
for c in none anon_high anon_low; do echo "af3|$c|--swap_options|_swap" >> "$Q"; done
for c in none anon_high anon_low; do echo "af3|$c|--paraphrase|_para" >> "$Q"; done
echo "af3|hedge_plain||" >> "$Q"
log "queued $(wc -l < "$Q")"
pop(){ exec 9>"$LK"; flock 9; local l; l=$(head -1 "$Q"); [ -n "$l" ] && sed -i '1d' "$Q"; flock -u 9; exec 9>&-; echo "$l"; }
worker(){ local wid=$1 g=$2
  while :; do
    local job; job=$(pop); [ -z "$job" ] && break
    IFS='|' read -r m cond extra ftag <<< "$job"
    local pfx="nm_${m}" out n
    out="outputs/${pfx}_sound_${cond}${ftag}_v1.jsonl"
    for t in 1 2 3; do
      [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ] && { n=400; break; }
      while :; do f=$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits); [ "$f" -gt 28000 ] && break; sleep 60; done
      if [ "$m" = af3 ]; then
        CUDA_VISIBLE_DEVICES=$g bash scripts/run_af3_v1.sh --attr sound --cond "$cond" $extra > "outputs/${pfx}_sound_${cond}${ftag}.log" 2>&1
      else
        CUDA_VISIBLE_DEVICES=$g python3 scripts/eval_newmodels_v1.py --model "$m" --attr sound --cond "$cond" $extra > "outputs/${pfx}_sound_${cond}${ftag}.log" 2>&1
      fi
      n=$(wc -l < "$out" 2>/dev/null||echo 0); [ "$n" -ge 400 ] && break; sleep 20
    done
    log "w$wid ${pfx}/${cond}${ftag} -> $n"
  done; log "worker $wid done"; }
worker 1 0 & worker 2 1 & worker 3 2 & worker 4 3 &
wait
log "GAP FILL ALL DONE"
