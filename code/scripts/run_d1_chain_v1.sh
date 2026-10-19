#!/bin/bash
set -u
CT2=/workspace/svhalluc/claude_track2
cd "$CT2"
# file-based wait: after C1b VGG output completes
until [ -f outputs/eval_vggsparse_qwen_inject_t3adapter_noav_v1.jsonl ] && [ "$(wc -l < outputs/eval_vggsparse_qwen_inject_t3adapter_noav_v1.jsonl)" -ge 795 ]; do sleep 300; done
CUDA_VISIBLE_DEVICES=1 python3 scripts/eval_vggsparse_direction_d1_v1.py --mode base \
  > outputs/eval_vggsparse_direction_base_v1.log 2>&1
echo "[d1] base done"
CUDA_VISIBLE_DEVICES=1 python3 scripts/eval_vggsparse_direction_d1_v1.py --mode inject \
  > outputs/eval_vggsparse_direction_inject_v1.log 2>&1
echo "[d1] inject done — D1 DONE"
