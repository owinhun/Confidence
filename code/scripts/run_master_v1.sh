#!/bin/bash
# Every remaining Nemotron condition, in one queue, drained by one worker per free card.
#
# Ordered by what the paper needs soonest:
#   1. null_block      the empty column in the one-step decomposition table
#   2. verbal ladder   the five levels, true and false, for the beta curve and pi* ladder
#                      (v_low and v_high already exist as anon_low / anon_high, and
#                       true_low / true_high are the ladder's true branch at those two rungs)
#   3. controls        epistemic synonyms and the expert source factor
#
# Workers hold a card for one condition at a time; Nemotron is 62 GB so a card takes exactly
# one. The queue is flock'd because two workers appending to one jsonl has bitten this before.
set -u
cd /workspace/svhalluc/claude_track2
S=outputs/master_status.txt
Q=/root/.claude/jobs/65f6dcfb/tmp/master_queue
LK=/root/.claude/jobs/65f6dcfb/tmp/master.lock
K(){ python3 -c "import datetime;print((datetime.datetime.utcnow()+datetime.timedelta(hours=9)).strftime('%m-%d %H:%M'))"; }
log(){ echo "[KST $(K)] $*" >> "$S"; }

: > "$Q"
for A in sound colour; do
  echo "$A|null_block|"        >> "$Q"
  for C in v_vlow v_mod v_vhigh; do
    echo "$A|$C|"              >> "$Q"
    echo "$A|$C|--claim true"  >> "$Q"
  done
  for C in expert_high expert_low reliab_high reliab_low certain_high certain_low; do
    echo "$A|$C|"              >> "$Q"
  done
done
log "queued $(wc -l < "$Q") conditions"

pop(){ exec 9>"$LK"; flock 9; local l; l=$(head -1 "$Q"); [ -n "$l" ] && sed -i '1d' "$Q"; flock -u 9; exec 9>&-; echo "$l"; }

worker(){ local wid=$1 g=$2
  while :; do
    local job; job=$(pop); [ -z "$job" ] && break
    IFS='|' read -r attr cond extra <<< "$job"
    local sfx=""; [[ "$extra" == *"claim true"* ]] && sfx="_T"
    local out="outputs/nm_nemotron_${attr}_${cond}${sfx}_v1.jsonl"
    # A "done" check on line count alone cannot see a run that is still in progress: the file
    # exists, is under 400 rows, and a second worker happily starts the same condition and
    # appends into it. Treat a file written within the last three minutes as owned by a live
    # worker and skip it.
    local fresh=0
    if [ -f "$out" ]; then
      local age=$(( $(date +%s) - $(stat -c %Y "$out") ))
      [ "$age" -lt 180 ] && fresh=1
    fi
    if [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ] || [ "$fresh" = 1 ]; then log "w$wid $attr/$cond$sfx skip (done)"; continue; fi
    while :; do f=$(nvidia-smi -i "$g" --query-gpu=memory.free --format=csv,noheader,nounits)
      [ "$f" -gt 65000 ] && break; sleep 60; done
    CUDA_VISIBLE_DEVICES=$g timeout 10800 bash scripts/run_nemotron_v1.sh \
      --attr "$attr" --cond "$cond" $extra > "${out%_v1.jsonl}.log" 2>&1
    log "w$wid $attr/$cond$sfx -> $(wc -l < "$out" 2>/dev/null || echo 0)"
  done; log "worker $wid done"; }

# Workers start immediately. Waiting for every earlier job to finish left two 80 GB cards
# idle while two null_block runs finished, which is exactly the stall this queue exists to
# avoid: the per-worker free-memory guard below already prevents two jobs landing on one card.
log "master queue starting; workers gate on free memory, not on other jobs"
worker 1 0 & worker 2 1 & worker 3 2 & worker 4 3 &
wait
log "MASTER QUEUE COMPLETE"
