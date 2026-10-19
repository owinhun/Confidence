#!/bin/bash
set -u
cd /workspace/svhalluc/claude_track2
S=outputs/crossprop_status.txt
log(){ echo "[$(TZ=Asia/Seoul date +%H:%M)] $*" >> "$S"; }
for c in none anon_high anon_low; do
  out="outputs/nm_qwen2audio_aux_noise_${c}_v1.jsonl"
  for t in 1 2 3; do
    [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ] && break
    while :; do f=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits); [ "$f" -gt 27000 ] && break; sleep 60; done
    CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_newmodels_v1.py --model qwen2audio \
      --attr aux_noise --cond "$c" > "outputs/nm_qwen2audio_aux_noise_${c}.log" 2>&1
    n=$(wc -l < "$out" 2>/dev/null||echo 0); log "q2a/$c try$t -> $n"; [ "$n" -ge 400 ] && break
  done
done
log "Q2A CROSSPROP DONE"
