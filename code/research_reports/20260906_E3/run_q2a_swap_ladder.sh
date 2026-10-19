#!/bin/bash
# E3: fill the missing swapped-order ladder rungs for Qwen2-Audio/sound. Runner unchanged.
set -u
cd /workspace/svhalluc/claude_track2
S=research_reports/20260906_E3/q2a_status.txt
log(){ echo "[$(date -u +%H:%M)Z] $*" | tee -a "$S"; }
pick_gpu(){ nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
  | awk -F', ' '$1<4 {free=$3-$2; if (free>26000) print free, $1}' | sort -rn | head -1 | awk '{print $2}'; }
for cond in v_low v_mod v_vhigh; do for claim in false true; do
  tag=""; [ "$claim" = true ] && tag="_T"
  out=outputs/nm_qwen2audio_sound_${cond}${tag}_swap_v1.jsonl
  n=0; [ -f "$out" ] && n=$(wc -l < "$out"); [ "$n" -ge 400 ] && { log "skip $cond $claim ($n)"; continue; }
  for attempt in 1 2 3; do
    g=""; while [ -z "$g" ]; do g=$(pick_gpu); [ -z "$g" ] && sleep 120; done
    log "q2a $cond $claim attempt $attempt GPU $g"
    CUDA_VISIBLE_DEVICES=$g PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 scripts/eval_newmodels_v1.py \
      --model qwen2audio --attr sound --cond $cond --claim $claim --swap_options \
      > outputs/nm_qwen2audio_sound_${cond}${tag}_swap.log 2>&1
    n=0; [ -f "$out" ] && n=$(wc -l < "$out")
    if [ "$n" -ge 400 ]; then log "q2a $cond $claim OK ($n)"; break; fi
    log "q2a $cond $claim incomplete ($n), retry"; sleep 120
  done
done; done
log "Q2A CHAIN DONE"
