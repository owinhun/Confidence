#!/bin/bash
set -u
CT2=/workspace/svhalluc/claude_track2; cd "$CT2"
S=/root/.claude/jobs/65f6dcfb/tmp/vgs_status.txt; : > "$S"
log(){ echo "[$(date -u +%H:%M)] $*" | tee -a "$S"; }
CUDA_VISIBLE_DEVICES=0 python3 scripts/synchformer_vggsoundsync_v1.py > outputs/sf_vgs.log 2>&1
log "expert scoring done: $(wc -l < outputs/synchformer_vggsoundsync_v1.jsonl 2>/dev/null || echo 0)"
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_vggsoundsync_qwen_v1.py --mode base > outputs/vgs_base.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python3 scripts/eval_vggsoundsync_qwen_v1.py --mode inject --binary --suffix _verdict > outputs/vgs_verdict.log 2>&1 &
wait
log "VGS CHAIN TERMINATED"
