#!/bin/bash
# The backlog: everything agreed for after the Nemotron master queue, in one list, dispatched
# by harness so a single set of workers can drain it.
#
#   A. Nemotron's remaining controls, so the new backbone has the same control suite as the
#      five it is being compared against -- otherwise Sec. 6 can only speak for Qwen cells.
#   B. Nemotron's numeric ladder, the second confidence scale.
#   C. The swapped-option half of every primary margin condition on the original five models.
#      Table 2 is already order-averaged; Table 1 and the beta ratio are still single-order,
#      which is the reviewer's P0-3 and the one place the option-order caveat still bites.
#
# It starts once the master queue file is empty. That barrier is deliberate: a Nemotron run
# needs ~62 GB of an 80 GB card, so two runners probing free memory at the same instant can
# both decide a card is free and OOM. Waiting on the queue file, not on running jobs, keeps the
# idle window to the tail of the last four master conditions.
set -u
cd /workspace/svhalluc/claude_track2
S=outputs/backlog_status.txt
Q=/root/.claude/jobs/65f6dcfb/tmp/backlog_queue
LK=/root/.claude/jobs/65f6dcfb/tmp/backlog.lock
K(){ python3 -c "import datetime;print((datetime.datetime.utcnow()+datetime.timedelta(hours=9)).strftime('%m-%d %H:%M'))"; }
log(){ echo "[KST $(K)] $*" >> "$S"; }

: > "$Q"
# A + B -- Nemotron parity with the existing control suite
for A in sound colour; do
  for C in irrelevant_high irrelevant_low bare_false hedge_plain otherclip_high otherclip_low; do
    echo "nemotron|$A|$C|" >> "$Q"; done
  for C in n_015 n_035 n_060 n_085 n_095; do echo "nemotron|$A|$C|" >> "$Q"; done
done
# C -- the swapped half of the primary margin conditions on the original five
for M in "s2 sound" "s2q3 sound" "s2vl colour" "qwen2audio sound" "af3 sound"; do
  set -- $M
  for C in priority_high priority_low srconly_high srconly_low priority_true_high priority_true_low true_high; do
    echo "$1|$2|$C|--swap_options" >> "$Q"; done
done
log "queued $(wc -l < "$Q") conditions (Nemotron parity + order-averaging)"

pop(){ exec 9>"$LK"; flock 9; local l; l=$(head -1 "$Q"); [ -n "$l" ] && sed -i '1d' "$Q"; flock -u 9; exec 9>&-; echo "$l"; }

outfile(){ local k=$1 a=$2 c=$3 e=$4 sfx=""
  [[ "$e" == *swap* ]] && sfx="_swap"; [[ "$e" == *"claim true"* ]] && sfx="_T$sfx"
  case "$k" in
    nemotron)   echo "outputs/nm_nemotron_${a}_${c}${sfx}_v1.jsonl";;
    s2)         echo "outputs/s2_${a}_${c}${sfx}_v1.jsonl";;
    s2q3)       echo "outputs/s2q3_${a}_${c}${sfx}_v1.jsonl";;
    s2vl)       echo "outputs/s2vl_${a}_${c}${sfx}_v1.jsonl";;
    qwen2audio) echo "outputs/nm_qwen2audio_${a}_${c}${sfx}_v1.jsonl";;
    af3)        echo "outputs/nm_af3_${a}_${c}${sfx}_v1.jsonl";;
  esac; }

run(){ local k=$1 a=$2 c=$3 e=$4 g=$5 lg=$6
  case "$k" in
    nemotron)   CUDA_VISIBLE_DEVICES=$g bash scripts/run_nemotron_v1.sh --attr "$a" --cond "$c" $e > "$lg" 2>&1;;
    s2)         CUDA_VISIBLE_DEVICES=$g python3 scripts/eval_stage2_factorial_v1.py --attr "$a" --cond "$c" $e > "$lg" 2>&1;;
    s2q3)       CUDA_VISIBLE_DEVICES=$g python3 scripts/eval_stage2_q3_v1.py --attr "$a" --cond "$c" $e > "$lg" 2>&1;;
    s2vl)       CUDA_VISIBLE_DEVICES=$g python3 scripts/eval_stage2_vllama_v1.py --attr "$a" --cond "$c" $e > "$lg" 2>&1;;
    qwen2audio) CUDA_VISIBLE_DEVICES=$g python3 scripts/eval_newmodels_v1.py --model qwen2audio --attr "$a" --cond "$c" $e > "$lg" 2>&1;;
    af3)        CUDA_VISIBLE_DEVICES=$g bash scripts/run_af3_v1.sh --attr "$a" --cond "$c" $e > "$lg" 2>&1;;
  esac; }

worker(){ local wid=$1 g=$2
  while :; do
    local job; job=$(pop); [ -z "$job" ] && break
    IFS='|' read -r kind attr cond extra <<< "$job"
    local out; out=$(outfile "$kind" "$attr" "$cond" "$extra")
    # A "done" check on line count alone cannot see a run that is still in progress: the file
    # exists, is under 400 rows, and a second worker happily starts the same condition and
    # appends into it. Treat a file written within the last three minutes as owned by a live
    # worker and skip it.
    local fresh=0
    if [ -f "$out" ]; then
      local age=$(( $(date +%s) - $(stat -c %Y "$out") ))
      [ "$age" -lt 180 ] && fresh=1
    fi
    [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ] || [ "$fresh" = 1 ] && { log "w$wid $kind/$attr/$cond skip"; continue; }
    # Nemotron needs a whole card; the 7B harnesses need ~25 GB
    local need=30000; [ "$kind" = nemotron ] && need=65000
    while :; do f=$(nvidia-smi -i "$g" --query-gpu=memory.free --format=csv,noheader,nounits)
      [ "$f" -gt "$need" ] && break; sleep 60; done
    run "$kind" "$attr" "$cond" "$extra" "$g" "${out%_v1.jsonl}.log"
    log "w$wid $kind/$attr/$cond${extra:+ $extra} -> $(wc -l < "$out" 2>/dev/null || echo 0)"
  done; log "worker $wid done"; }

while [ -s /root/.claude/jobs/65f6dcfb/tmp/master_queue ]; do sleep 120; done
log "master queue drained; backlog starting"
worker 1 0 & worker 2 1 & worker 3 2 & worker 4 3 &
wait
log "BACKLOG COMPLETE"
