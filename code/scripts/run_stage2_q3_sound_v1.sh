#!/bin/bash
# Qwen3-Omni Stage-2 replication, SOUND FIRST.
# Sound is the only attribute that survives every control on Qwen2.5 (greyscale is at
# ceiling, silence fails the irrelevant-statement control), so replicating the confidence
# effect there is what removes the "one model" limitation. Colour controls follow.
set -u
CT2=/workspace/svhalluc/claude_track2; cd "$CT2"
S=/root/.claude/jobs/65f6dcfb/tmp/s2q3_sound_status.txt
log(){ echo "[$(date -u +%H:%M)] $*" | tee -a "$S"; }

wait_for_gpu(){  # $1 gpu id -- Qwen3-30B needs ~76 GiB, so demand a nearly empty card
  local g=$1
  while :; do
    local used; used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$used" -lt 3000 ] && return 0
    sleep 120
  done
}

run(){  # $1 gpu  $2 attr  $3 cond
  local g=$1 a=$2 c=$3
  local out="outputs/s2q3_${a}_${c}_v1.jsonl"
  for try in 1 2 3; do
    [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ] && { log "q3 $a/$c already complete"; return 0; }
    wait_for_gpu "$g"
    CUDA_VISIBLE_DEVICES=$g python3 scripts/eval_stage2_q3_v1.py --attr "$a" --cond "$c" \
      > "outputs/s2q3_${a}_${c}.log" 2>&1
    local n; n=$(wc -l < "$out" 2>/dev/null || echo 0)
    log "q3 $a/$c try$try -> $n"
    [ "$n" -ge 400 ] && return 0
    sleep 30
  done
  log "q3 $a/$c FAILED after 3 tries"
}

# three independent chains, one per free card
( run 0 sound none; run 0 sound anon_high;   run 0 sound true_high;        log "CHAIN0 DONE" ) &
( run 1 sound anon_low; run 1 sound expert_high; run 1 sound irrelevant_high; log "CHAIN1 DONE" ) &
( run 2 sound expert_low; run 2 colour true_high; run 2 colour irrelevant_high; log "CHAIN2 DONE" ) &
wait
log "STAGE2-Q3-SOUND ALL DONE"
