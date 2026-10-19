#!/bin/bash
set -u
CT2=/workspace/svhalluc/claude_track2
cd "$CT2"
# queue after the controls chain
while pgrep -f run_controls_chain_v1.sh > /dev/null; do sleep 180; done
CUDA_VISIBLE_DEVICES=1 python3 scripts/expert_inject_ta_t2_v1.py --binary \
  --out outputs/expert_inject_ta_binary_v1.jsonl > outputs/expert_inject_ta_binary_v1.log 2>&1
echo "[abl] A1 binary done"
CUDA_VISIBLE_DEVICES=1 python3 scripts/expert_inject_ta_t2_v1.py --k1 \
  --out outputs/expert_inject_ta_k1_v1.jsonl > outputs/expert_inject_ta_k1_v1.log 2>&1
echo "[abl] A2 k1 done — ABLATIONS DONE"
