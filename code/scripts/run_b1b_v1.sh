#!/bin/bash
set -u
CT2=/workspace/svhalluc/claude_track2
cd /workspace/svhalluc
# GPU0: wait for A1-VGG binary retry then overfit control to be done (both file-based)
until [ -f $CT2/outputs/eval_vggsparse_qwen_inject_binary_v1.jsonl ] && [ "$(wc -l < $CT2/outputs/eval_vggsparse_qwen_inject_binary_v1.jsonl)" -ge 795 ]; do sleep 300; done
until [ -f $CT2/outputs/train_lora_overfit16_v1.log ] && grep -q "epoch 40\|DONE\|\[val\] epoch 4" $CT2/outputs/train_lora_overfit16_v1.log; do
  if ! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then break; fi; sleep 300; done
CUDA_VISIBLE_DEVICES=0 python3 $CT2/scripts/vllama_inject_ta_b1_v1.py --mode inject --binary \
  > $CT2/outputs/vllama_inject_ta_b1_inject_binary_v1.log 2>&1
echo "[b1b] vllama verdict-wording inject done"
