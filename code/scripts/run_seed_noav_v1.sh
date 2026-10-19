#!/bin/bash
set -u
CT2=/workspace/svhalluc/claude_track2; cd "$CT2"
S=/root/.claude/jobs/65f6dcfb/tmp/seed_noav_status.txt; : > "$S"
log(){ echo "[$(date -u +%H:%M)] $*" | tee -a "$S"; }
for pair in "1 0" "2 1"; do
  set -- $pair; seed=$1; gpu=$2
  ( CUDA_VISIBLE_DEVICES=$gpu python3 scripts/expert_inject_ta_t2_v1.py --no_av \
      --adapter outputs/lora_evidence_s${seed}/epoch2 \
      --out outputs/expert_inject_ta_s${seed}_noav_v1.jsonl \
      > outputs/expert_inject_ta_s${seed}_noav.log 2>&1
    log "seed $seed no-AV done" ) &
done
wait; log "SEED-NOAV TERMINATED"
