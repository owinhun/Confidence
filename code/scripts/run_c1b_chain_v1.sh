#!/bin/bash
# C1b: adapter + inject WITHOUT AV — decisive fusion test for T3.
set -u
CT2=/workspace/svhalluc/claude_track2
cd "$CT2"
ADAPTER=outputs/lora_evidence_v1/epoch2
# wait for GPU1 ablation runs (file-based check, no pgrep self-match)
until [ -f outputs/expert_inject_ta_k1_v1.jsonl ] && [ "$(wc -l < outputs/expert_inject_ta_k1_v1.jsonl)" -ge 320 ]; do sleep 180; done
CUDA_VISIBLE_DEVICES=1 python3 scripts/expert_inject_ta_t2_v1.py --no_av --adapter "$ADAPTER" \
  --out outputs/expert_inject_ta_t3adapter_noav_v1.jsonl > outputs/expert_inject_ta_t3adapter_noav_v1.log 2>&1
echo "[c1b] TA adapter no_av done"
CUDA_VISIBLE_DEVICES=1 python3 scripts/eval_vggsparse_qwen_v1.py --mode inject --no_av \
  --adapter "$ADAPTER" --suffix _t3adapter_noav > outputs/eval_vggsparse_qwen_inject_t3adapter_noav_v1.log 2>&1
echo "[c1b] VGG adapter no_av done — C1b DONE"
