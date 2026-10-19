#!/bin/bash
set -u
source /workspace/svhalluc/claude_track2/scripts/gpu_queue_v1.sh
: > "$S"
log "QUEUE2 START"
run_job r1_qwen3omni_retry 70000 outputs/qwen3omni_r1_base_v1.jsonl 180 \
  python3 scripts/eval_qwen3omni_r1_v1.py --limit 200
log "QUEUE2 TERMINATED"
