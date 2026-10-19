#!/bin/bash
set -u
CT2=/workspace/svhalluc/claude_track2
cd /workspace/svhalluc
# wait for B1 base (file count) then run B1 inject, then A1-VGG binary
until [ -f $CT2/outputs/vllama_inject_ta_b1_base_v1.jsonl ] && [ "$(wc -l < $CT2/outputs/vllama_inject_ta_b1_base_v1.jsonl)" -ge 320 ]; do sleep 300; done
CUDA_VISIBLE_DEVICES=0 python3 $CT2/scripts/vllama_inject_ta_b1_v1.py --mode inject \
  > $CT2/outputs/vllama_inject_ta_b1_inject_v1.log 2>&1
echo "[gpu0] B1 inject done"
CUDA_VISIBLE_DEVICES=0 python3 $CT2/scripts/eval_vggsparse_qwen_v1.py --mode inject --binary \
  --suffix _binary > $CT2/outputs/eval_vggsparse_qwen_inject_binary_v1.log 2>&1
echo "[gpu0] A1-VGG binary done — GPU0 QUEUE DONE"
