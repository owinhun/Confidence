#!/bin/bash
set -u
CT2=/workspace/svhalluc/claude_track2
cd /workspace/svhalluc
# queue after ablations (which queue after controls) — match on script FILE name only
while pgrep -f "run_ablations_chain_v1" | grep -qv $$; do sleep 180; done 2>/dev/null
while pgrep -f "expert_inject_ta_t2_v1.py" > /dev/null; do sleep 180; done
CUDA_VISIBLE_DEVICES=1 python3 $CT2/scripts/vllama_inject_ta_b1_v1.py --mode base \
  > $CT2/outputs/vllama_inject_ta_b1_base_v1.log 2>&1
echo "[b1] base done"
CUDA_VISIBLE_DEVICES=1 python3 $CT2/scripts/vllama_inject_ta_b1_v1.py --mode inject \
  > $CT2/outputs/vllama_inject_ta_b1_inject_v1.log 2>&1
echo "[b1] inject done — B1 DONE"
