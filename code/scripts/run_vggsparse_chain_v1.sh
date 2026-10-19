#!/bin/bash
# Orchestrates the VGGSound-Sparse eval pipeline after pair build completes.
set -u
CT2=/workspace/svhalluc/claude_track2
cd "$CT2"

# 1. wait for pair build
while pgrep -f build_vggsparse_pairs_v2.py > /dev/null; do sleep 60; done
echo "[chain] build done: $(wc -l < data/vggsparse_pairs_v2.jsonl) clips"

# 2. wait for GPU0 pairs_all scoring to finish, then score vggsparse pairs on GPU0
while pgrep -f synchformer_pairs_all_v1.py > /dev/null; do sleep 60; done
CUDA_VISIBLE_DEVICES=0 python3 scripts/synchformer_vggsparse_v1.py \
  > outputs/synchformer_vggsparse_v1.log 2>&1 &
SCORE_PID=$!

# 3. Qwen baseline on GPU1 (independent of expert scores)
CUDA_VISIBLE_DEVICES=1 python3 scripts/eval_vggsparse_qwen_v1.py --mode base \
  > outputs/eval_vggsparse_qwen_base_v1.log 2>&1
echo "[chain] qwen base done"

# 4. wait for expert scoring, then Qwen inject on GPU1
wait $SCORE_PID
echo "[chain] expert scoring done: $(wc -l < outputs/synchformer_vggsparse_v1.jsonl)"
CUDA_VISIBLE_DEVICES=1 python3 scripts/eval_vggsparse_qwen_v1.py --mode inject \
  > outputs/eval_vggsparse_qwen_inject_v1.log 2>&1
echo "[chain] qwen inject done — ALL DONE"
