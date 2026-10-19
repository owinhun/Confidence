#!/bin/bash
set -u
CT2=/workspace/svhalluc/claude_track2
cd "$CT2"
until [ -f outputs/expert_inject_ta_adapter_perm_noav_v1.jsonl ] && [ "$(wc -l < outputs/expert_inject_ta_adapter_perm_noav_v1.jsonl)" -ge 320 ]; do sleep 300; done
CUDA_VISIBLE_DEVICES=1 python3 scripts/eval_vggsparse_direction_d1_v1.py --mode inject --binary \
  --suffix _binary > outputs/eval_vggsparse_direction_inject_binary_v1.log 2>&1
echo "[d1b] direction verdict-wording done"
