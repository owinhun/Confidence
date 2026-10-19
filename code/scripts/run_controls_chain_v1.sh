#!/bin/bash
set -u
CT2=/workspace/svhalluc/claude_track2
cd "$CT2"
# wait for any GPU1 job (frozen TA baseline) to finish
while pgrep -f eval_lora_ta_v1 > /dev/null; do sleep 120; done

CUDA_VISIBLE_DEVICES=1 python3 scripts/expert_inject_ta_t2_v1.py --no_av \
  --out outputs/expert_inject_ta_noav_v1.jsonl > outputs/expert_inject_ta_noav_v1.log 2>&1
echo "[controls] TA no_av done"
CUDA_VISIBLE_DEVICES=1 python3 scripts/eval_vggsparse_qwen_v1.py --mode inject --no_av \
  --suffix _noav > outputs/eval_vggsparse_qwen_inject_noav_v1.log 2>&1
echo "[controls] VGG no_av done"
CUDA_VISIBLE_DEVICES=1 python3 scripts/expert_inject_ta_t2_v1.py --permute \
  --out outputs/expert_inject_ta_perm_v1.jsonl > outputs/expert_inject_ta_perm_v1.log 2>&1
echo "[controls] TA permute done"
CUDA_VISIBLE_DEVICES=1 python3 scripts/eval_vggsparse_qwen_v1.py --mode inject --permute \
  --suffix _perm > outputs/eval_vggsparse_qwen_inject_perm_v1.log 2>&1
echo "[controls] VGG permute done — CONTROLS ALL DONE"
