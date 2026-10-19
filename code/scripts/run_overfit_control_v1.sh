#!/bin/bash
set -u
CT2=/workspace/svhalluc/claude_track2
cd "$CT2"
# GPU0: wait for A1-VGG binary retry to complete (file-based)
until [ -f outputs/eval_vggsparse_qwen_inject_binary_v1.jsonl ] && [ "$(wc -l < outputs/eval_vggsparse_qwen_inject_binary_v1.jsonl)" -ge 795 ]; do sleep 300; done
CUDA_VISIBLE_DEVICES=0 python3 scripts/train_lora_sync_v1.py \
  --data data/overfit16_v1.jsonl --fallback_data data/overfit16_v1.jsonl \
  --out_dir outputs/lora_overfit16_v1 --epochs 40 --grad_accum 4 --lr 1e-4 \
  --schedule constant --log_every 4 > outputs/train_lora_overfit16_v1.log 2>&1
echo "[posctl] 16-sample overfit run done"
