#!/bin/bash
# E1 master chain — runs unattended. File-count based waits only (no pgrep).
set -u
CT2=/workspace/svhalluc/claude_track2
cd "$CT2"
log(){ echo "[$(date -u +%H:%M)] $*"; }

# 1. wait for swap-clip build to finish (403 temporal + ~403 semantic rows)
until [ -f data/swap_pairs_v1.jsonl ] && [ "$(wc -l < data/swap_pairs_v1.jsonl)" -ge 790 ]; do sleep 120; done
log "swap build done: $(wc -l < data/swap_pairs_v1.jsonl) rows"

# 2. expert scoring on all swap-task clips (GPU 0)
CUDA_VISIBLE_DEVICES=0 python3 scripts/synchformer_swap_v1.py > outputs/synchformer_swap_v1.log 2>&1
log "expert scoring done: $(wc -l < outputs/synchformer_swap_v1.jsonl)"

# 3. three Qwen conditions in parallel on GPUs 1/2/3
CUDA_VISIBLE_DEVICES=1 python3 scripts/eval_why_task_v1.py --mode base    > outputs/why_task_base_v1.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=2 python3 scripts/eval_why_task_v1.py --mode verdict > outputs/why_task_verdict_v1.log 2>&1 &
P2=$!
CUDA_VISIBLE_DEVICES=3 python3 scripts/eval_why_task_v1.py --mode prob    > outputs/why_task_prob_v1.log 2>&1 &
P3=$!
wait $P1 $P2 $P3
log "all three Qwen conditions done"

# 4. adapter condition on GPU 0 (probability evidence + evidence-use LoRA)
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_why_task_v1.py --mode prob --tag _adapter \
  --adapter outputs/lora_evidence_v1/epoch2 > outputs/why_task_prob_adapter_v1.log 2>&1
log "E1 CHAIN COMPLETE"
