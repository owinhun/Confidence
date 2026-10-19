#!/bin/bash
# GPU3 chain: the format-effect + engagement-control replication on Qwen3.
set -u
CT2=/workspace/svhalluc/claude_track2; cd "$CT2"
S=/root/.claude/jobs/65f6dcfb/tmp/s2q3_sound_status.txt
log(){ echo "[$(date -u +%H:%M)] $*" | tee -a "$S"; }
wait_for_gpu(){ while :; do u=$(nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 3000 ] && return 0; sleep 120; done; }
run(){ local g=$1 a=$2 c=$3 extra="${4:-}" tag="${5:-}"
  local out="outputs/s2q3_${a}_${c}${tag}_v1.jsonl"
  for try in 1 2 3; do
    [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ] && { log "q3 $a/$c$tag complete"; return 0; }
    wait_for_gpu "$g"
    CUDA_VISIBLE_DEVICES=$g python3 scripts/eval_stage2_q3_v1.py --attr "$a" --cond "$c" $extra \
      > "outputs/s2q3_${a}_${c}${tag}.log" 2>&1
    local n; n=$(wc -l < "$out" 2>/dev/null || echo 0); log "q3 $a/$c$tag try$try -> $n"
    [ "$n" -ge 400 ] && return 0; sleep 30
  done; log "q3 $a/$c$tag FAILED"; }
run 3 sound  bare_false
run 3 colour bare_false
run 3 sound  anon_high "--mask_sensory --tag _masked" "_masked"
run 3 colour anon_high "--mask_sensory --tag _masked" "_masked"
log "CHAIN3 DONE"
