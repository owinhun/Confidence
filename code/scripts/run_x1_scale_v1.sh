#!/bin/bash
# X1: LoRA unlearnability at 10x data scale (2,317 VGGSound clips, no evidence in prompt).
set -u
CT2=/workspace/svhalluc/claude_track2; cd "$CT2"
S=/root/.claude/jobs/65f6dcfb/tmp/x1_status.txt; : > "$S"
log(){ echo "[$(date -u +%H:%M)] $*" | tee -a "$S"; }
log "X1 training start (2317 clips, r=16, lr 5e-5, 2 epochs)"
CUDA_VISIBLE_DEVICES=0 python3 scripts/train_lora_sync_v1.py \
  --data data/vggtrain_pairs_v1.jsonl --fallback_data data/vggtrain_pairs_v1.jsonl \
  --out_dir outputs/lora_scale_x1 --lr 5e-5 --schedule constant --epochs 2 --val_limit 150 \
  > outputs/train_lora_scale_x1.log 2>&1
log "training done; best val: $(grep -o 'acc=[0-9]*/[0-9]*=[0-9.]*%' outputs/train_lora_scale_x1.log | tail -2 | tr '\n' ' ')"
A=outputs/lora_scale_x1/epoch2
[ -d "$A" ] || A=outputs/lora_scale_x1/epoch1
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_vggsparse_qwen_v1.py --mode base --adapter "$A" \
  --suffix _scalex1 > outputs/eval_vggs_scalex1.log 2>&1
log "VGGS eval: $(wc -l < outputs/eval_vggsparse_qwen_base_scalex1_v1.jsonl 2>/dev/null || echo 0) rows"
log "X1 TERMINATED"
