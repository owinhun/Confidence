#!/bin/bash
# Memory-aware job queue for GPUs 0-3 (never 4-7).
# Waits for a GPU with enough FREE memory before launching, so it yields to the
# user's own jobs instead of crowding them out. Retries on failure. Always
# writes a terminal status line so watchers detect completion AND failure.
set -u
CT2=/workspace/svhalluc/claude_track2
cd "$CT2"
S=/root/.claude/jobs/65f6dcfb/tmp/queue_status.txt
log(){ echo "[$(date -u +%H:%M)] $*" | tee -a "$S"; }

# $1 = MiB required; echoes a gpu index once one is free enough, else waits
wait_for_gpu(){
  local need=$1
  while true; do
    local g
    g=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
        | awk -F', ' -v n="$need" '$1<4 {free=$3-$2; if (free>n+6000) print free, $1}' \
        | sort -rn | head -1 | awk '{print $2}')
    if [ -n "$g" ]; then echo "$g"; return; fi
    sleep 120
  done
}

# run_job <label> <need_MiB> <outfile> <min_rows> <cmd...>
run_job(){
  local label=$1 need=$2 outf=$3 minrows=$4; shift 4
  for attempt in 1 2 3; do
    local g; g=$(wait_for_gpu "$need")
    log "$label attempt $attempt on GPU $g"
    CUDA_VISIBLE_DEVICES=$g "$@" > "outputs/${label}.log" 2>&1
    local n=0; [ -f "$outf" ] && n=$(wc -l < "$outf")
    if [ "$n" -ge "$minrows" ]; then log "$label OK ($n rows)"; return 0; fi
    log "$label incomplete ($n/$minrows), retrying in 180s"
    sleep 180
  done
  log "$label FAILED"
  return 1
}
