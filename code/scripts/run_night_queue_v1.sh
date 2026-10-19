#!/bin/bash
set -u
source /workspace/svhalluc/claude_track2/scripts/gpu_queue_v1.sh
: > "$S"
log "NIGHT QUEUE START"
# 1. R1 existential check: does the newer omni generation solve synchrony? (needs ~70 GB)
run_job r1_qwen3omni 70000 outputs/qwen3omni_r1_base_v1.jsonl 180 \
  python3 scripts/eval_qwen3omni_r1_v1.py --limit 200
# 2. N2 expert scoring (per-half), then the three Qwen conditions
run_job n2_expert 12000 outputs/synchformer_n2_v1.jsonl 390 \
  python3 scripts/synchformer_n2_v1.py
run_job n2_base 26000 outputs/eval_n2_base_v1.jsonl 390 \
  python3 scripts/eval_n2_qwen_v1.py --mode base
run_job n2_verdict 26000 outputs/eval_n2_inject_verdict_v1.jsonl 385 \
  python3 scripts/eval_n2_qwen_v1.py --mode inject --binary --suffix _verdict
log "NIGHT QUEUE TERMINATED"
