#!/bin/bash
set -u
CT2=/workspace/svhalluc/claude_track2
cd "$CT2"
AD=outputs/lora_evidence_v1/epoch2
# GPU1: wait for D1 inject to complete (file-based)
until [ -f outputs/eval_vggsparse_direction_inject_v1.jsonl ] && [ "$(wc -l < outputs/eval_vggsparse_direction_inject_v1.jsonl)" -ge 395 ]; do sleep 300; done
CUDA_VISIBLE_DEVICES=1 python3 scripts/expert_inject_ta_t2_v1.py --adapter "$AD" --permute \
  --out outputs/expert_inject_ta_adapter_perm_av_v1.jsonl > outputs/expert_inject_ta_adapter_perm_av_v1.log 2>&1
echo "[gate] adapter+wrongEvid+AV done"
CUDA_VISIBLE_DEVICES=1 python3 scripts/expert_inject_ta_t2_v1.py --adapter "$AD" --permute --no_av \
  --out outputs/expert_inject_ta_adapter_perm_noav_v1.jsonl > outputs/expert_inject_ta_adapter_perm_noav_v1.log 2>&1
echo "[gate] adapter+wrongEvid+noAV done — WRONG-EVIDENCE 2x2 DONE"
