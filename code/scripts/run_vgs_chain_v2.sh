#!/bin/bash
set -u
CT2=/workspace/svhalluc/claude_track2; cd "$CT2"
S=/root/.claude/jobs/65f6dcfb/tmp/vgs_status.txt; : > "$S"
log(){ echo "[$(date -u +%H:%M)] $*" | tee -a "$S"; }
pick(){ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1<4 {print $2, $1}' | sort -n | head -1 | awk '{print $2}'; }
for a in 1 2 3; do
  g=$(pick); log "expert scoring attempt $a on GPU $g"
  CUDA_VISIBLE_DEVICES=$g python3 scripts/synchformer_vggsoundsync_v1.py > outputs/sf_vgs.log 2>&1
  n=$(wc -l < outputs/synchformer_vggsoundsync_v2.jsonl 2>/dev/null || echo 0)
  [ "$n" -ge 590 ] && { log "expert OK ($n)"; break; } || { log "expert incomplete ($n) retry"; sleep 120; }
done
CUDA_VISIBLE_DEVICES=$(pick) python3 scripts/eval_vggsoundsync_qwen_v1.py --mode base > outputs/vgs_base.log 2>&1 &
P1=$!
sleep 20
CUDA_VISIBLE_DEVICES=$(pick) python3 scripts/eval_vggsoundsync_qwen_v1.py --mode inject --binary --suffix _verdict > outputs/vgs_verdict.log 2>&1 &
P2=$!
wait $P1 $P2
log "base=$(wc -l < outputs/eval_vggsoundsync_base_v1.jsonl 2>/dev/null || echo 0) verdict=$(wc -l < outputs/eval_vggsoundsync_inject_verdict_v1.jsonl 2>/dev/null || echo 0)"
log "VGS CHAIN TERMINATED"
