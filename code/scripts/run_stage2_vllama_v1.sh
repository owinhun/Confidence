#!/bin/bash
# Third-model replication: VideoLLaMA2.1-7B-AV, same held-out manifests, same design.
set -u
CT2=/workspace/svhalluc/claude_track2; cd "$CT2"
S=/root/.claude/jobs/65f6dcfb/tmp/s2vl_status.txt
log(){ echo "[$(date -u +%H:%M)] $*" | tee -a "$S"; }
run(){ local g=$1 a=$2 c=$3 extra="${4:-}" tag="${5:-}"
  local out="outputs/s2vl_${a}_${c}${tag}_v1.jsonl"
  for try in 1 2 3; do
    [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ] && { log "vl $a/$c$tag complete"; return 0; }
    CUDA_VISIBLE_DEVICES=$g python3 scripts/eval_stage2_vllama_v1.py \
      --attr "$a" --cond "$c" $extra > "outputs/s2vl_${a}_${c}${tag}.log" 2>&1
    local n; n=$(wc -l < "$out" 2>/dev/null || echo 0); log "vl $a/$c$tag try$try -> $n"
    [ "$n" -ge 400 ] && return 0; sleep 20
  done; log "vl $a/$c$tag FAILED"; }

( for c in none anon_high anon_low expert_high expert_low; do run 0 sound $c; done; log "VL0 DONE" ) &
( for c in true_high irrelevant_high bare_false; do run 1 sound $c; done
  run 1 sound anon_high "--mask_sensory --tag _masked" "_masked"; log "VL1 DONE" ) &
( for c in none anon_high anon_low expert_high expert_low; do run 2 colour $c; done; log "VL2 DONE" ) &
( for c in true_high irrelevant_high bare_false; do run 3 colour $c; done
  run 3 colour anon_high "--mask_sensory --tag _masked" "_masked"; log "VL3 DONE" ) &
wait
log "STAGE2-VLLAMA ALL DONE"
