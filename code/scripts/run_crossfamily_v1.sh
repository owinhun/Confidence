#!/bin/bash
# Broaden the evidence beyond one organisation. Four of the five evaluable models are
# Alibaba-affiliated (Qwen2.5/Qwen3/Qwen2-Audio, and VideoLLaMA2.1 from Alibaba DAMO), so the
# controls that currently rest on three Qwen models need at least one independent family.
#   AF3   (NVIDIA)     -- the six controls it is missing
#   Phi-4 (Microsoft)  -- re-screened with swapped options: its recall 100/2 may be a
#                         position bias rather than an inability, exactly as VideoLLaMA's was
# GPUs 0-3 only; each job waits on free memory rather than assuming an idle card.
set -u
cd /workspace/svhalluc/claude_track2
S=outputs/crossfam_status.txt
Q=/root/.claude/jobs/65f6dcfb/tmp/crossfam_queue
LK=/root/.claude/jobs/65f6dcfb/tmp/crossfam.lock
K(){ python3 -c "import datetime;print((datetime.datetime.utcnow()+datetime.timedelta(hours=9)).strftime('%H:%M'))"; }
log(){ echo "[KST $(K)] $*" >> "$S"; }
: > "$Q"
for c in null_block priority_high priority_low n_055 n_065 n_075; do echo "af3|$c|" >> "$Q"; done
for c in none anon_high anon_low; do echo "phi4mm|$c|--swap_options" >> "$Q"; done
log "queued $(wc -l < "$Q")"
pop(){ exec 9>"$LK"; flock 9; local l; l=$(head -1 "$Q"); [ -n "$l" ] && sed -i '1d' "$Q"; flock -u 9; exec 9>&-; echo "$l"; }
worker(){ local wid=$1 g=$2
  while :; do
    local job; job=$(pop); [ -z "$job" ] && break
    IFS='|' read -r m cond extra <<< "$job"
    local ftag=""; [ -n "$extra" ] && ftag="_swap"
    local out="outputs/nm_${m}_sound_${cond}${ftag}_v1.jsonl" n=0
    for t in 1 2 3; do
      [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ] && { n=400; break; }
      while :; do f=$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits); [ "$f" -gt 28000 ] && break; sleep 90; done
      if [ "$m" = af3 ]; then
        CUDA_VISIBLE_DEVICES=$g bash scripts/run_af3_v1.sh --attr sound --cond "$cond" $extra > "outputs/nm_${m}_sound_${cond}${ftag}.log" 2>&1
      else
        CUDA_VISIBLE_DEVICES=$g python3 scripts/eval_newmodels_v1.py --model "$m" --attr sound --cond "$cond" $extra > "outputs/nm_${m}_sound_${cond}${ftag}.log" 2>&1
      fi
      n=$(wc -l < "$out" 2>/dev/null||echo 0); [ "$n" -ge 400 ] && break; sleep 20
    done
    log "w$wid ${m}/${cond}${ftag} -> $n"
  done; log "worker $wid done"; }
worker 1 0 & worker 2 1 & worker 3 2 & worker 4 3 &
wait
log "CROSS-FAMILY CONTROLS DONE"
