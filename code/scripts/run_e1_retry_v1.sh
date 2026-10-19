#!/bin/bash
# E1 recovery chain: OOM-resilient. Picks a free GPU, retries on failure,
# and always writes a terminal status line so a watcher can detect completion
# OR failure (never silent).
set -u
CT2=/workspace/svhalluc/claude_track2
cd "$CT2"
STATUS=/root/.claude/jobs/65f6dcfb/tmp/e1_status.txt
: > "$STATUS"
log(){ echo "[$(date -u +%H:%M)] $*" | tee -a "$STATUS"; }

pick_gpu(){  # echo index of the GPU with most free memory among 0-3
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' '$1<4 {print $2, $1}' | sort -n | head -1 | awk '{print $2}'
}

run_retry(){  # $1=label $2=logfile $3=outfile $4=min_rows  rest=command
  local label=$1 logf=$2 outf=$3 minrows=$4; shift 4
  for attempt in 1 2 3 4 5; do
    local g; g=$(pick_gpu)
    log "$label attempt $attempt on GPU $g"
    CUDA_VISIBLE_DEVICES=$g "$@" > "$logf" 2>&1
    local n=0; [ -f "$outf" ] && n=$(wc -l < "$outf")
    if [ "$n" -ge "$minrows" ]; then log "$label OK ($n rows)"; return 0; fi
    log "$label incomplete ($n/$minrows) — retrying in 180s"
    sleep 180
  done
  log "$label FAILED after 5 attempts"
  return 1
}

# 1. expert scoring on the swap-task clips (resumes if partially done)
run_retry "expert-scoring" outputs/synchformer_swap_v1.log \
  outputs/synchformer_swap_v1.jsonl 795 \
  python3 scripts/synchformer_swap_v1.py

# 2. the three frozen conditions, sequentially (each retries, each resumes)
run_retry "why-base" outputs/why_task_base_v1.log \
  outputs/why_task_base_v1.jsonl 795 \
  python3 scripts/eval_why_task_v1.py --mode base
run_retry "why-verdict" outputs/why_task_verdict_v1.log \
  outputs/why_task_verdict_v1.jsonl 780 \
  python3 scripts/eval_why_task_v1.py --mode verdict
run_retry "why-prob" outputs/why_task_prob_v1.log \
  outputs/why_task_prob_v1.jsonl 780 \
  python3 scripts/eval_why_task_v1.py --mode prob

# 3. adapter condition
run_retry "why-prob-adapter" outputs/why_task_prob_adapter_v1.log \
  outputs/why_task_prob_adapter_v1.jsonl 780 \
  python3 scripts/eval_why_task_v1.py --mode prob --tag _adapter \
  --adapter outputs/lora_evidence_v1/epoch2

log "E1 CHAIN TERMINATED"
