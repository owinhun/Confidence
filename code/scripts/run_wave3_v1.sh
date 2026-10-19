#!/bin/bash
# Wave 3. Two questions both reviews raised, and neither can be answered from existing runs.
#
#  A. Is any of this about the signal at all?  The confidence field arrives as text, so a
#     reviewer can ask whether this is ordinary text-only sycophancy that would happen with no
#     audio or video attached. --mask_sensory withholds the clip and leaves the prompt
#     identical, so beta computed with and without the signal answers it directly.
#
#  B. Is Priority a clean non-epistemic control?  "Priority: high" can be read as an
#     instruction to prioritise the claim, which is not the same as carrying no epistemic
#     content. Urgency is action-relevant in the same way; Metric-X is graded but bears on
#     nothing. If the effect were about any filled graded field, all three would match
#     Confidence.
set -u
cd /workspace/svhalluc/claude_track2
S=outputs/wave3_status.txt
Q=/root/.claude/jobs/65f6dcfb/tmp/wave3_queue
LK=/root/.claude/jobs/65f6dcfb/tmp/wave3.lock
K(){ python3 -c "import datetime;print((datetime.datetime.utcnow()+datetime.timedelta(hours=9)).strftime('%m-%d %H:%M'))"; }
log(){ echo "[KST $(K)] $*" >> "$S"; }

: > "$Q"
CELLS="s2:sound s2q3:sound s2vl:colour qwen2audio:sound af3:sound nemotron:sound nemotron:colour"
# B first -- cheaper, and it is the control the design rests on
for spec in $CELLS; do
  m=${spec%%:*}; a=${spec##*:}
  for c in urgency_high urgency_low opaque_high opaque_low; do echo "$m|$a|$c|" >> "$Q"; done
done
# A -- signal withheld, same prompt
for spec in $CELLS; do
  m=${spec%%:*}; a=${spec##*:}
  for c in none anon_high anon_low true_high true_low; do echo "$m|$a|$c|--mask_sensory" >> "$Q"; done
done
log "queued $(wc -l < "$Q") conditions"

pop(){ exec 9>"$LK"; flock 9; local l; l=$(head -1 "$Q" 2>/dev/null || true)
  [ -n "$l" ] && sed -i '1d' "$Q"; flock -u 9; exec 9>&-; printf '%s' "$l"; }
outfile(){ local k=$1 a=$2 c=$3 e=$4 sfx=""
  case "$e" in *mask*) sfx="_masked";; esac
  case "$k" in
    nemotron)   printf 'outputs/nm_nemotron_%s_%s%s_v1.jsonl' "$a" "$c" "$sfx";;
    s2)         printf 'outputs/s2_%s_%s%s_v1.jsonl' "$a" "$c" "$sfx";;
    s2q3)       printf 'outputs/s2q3_%s_%s%s_v1.jsonl' "$a" "$c" "$sfx";;
    s2vl)       printf 'outputs/s2vl_%s_%s%s_v1.jsonl' "$a" "$c" "$sfx";;
    qwen2audio) printf 'outputs/nm_qwen2audio_%s_%s%s_v1.jsonl' "$a" "$c" "$sfx";;
    af3)        printf 'outputs/nm_af3_%s_%s%s_v1.jsonl' "$a" "$c" "$sfx";;
  esac; }
run(){ local k=$1 a=$2 c=$3 e=$4 g=$5 lg=$6
  case "$k" in
    nemotron)   CUDA_VISIBLE_DEVICES=$g timeout 10800 bash scripts/run_nemotron_v1.sh --attr "$a" --cond "$c" $e > "$lg" 2>&1 || true;;
    s2)         CUDA_VISIBLE_DEVICES=$g timeout 10800 python3 scripts/eval_stage2_factorial_v1.py --attr "$a" --cond "$c" $e > "$lg" 2>&1 || true;;
    s2q3)       CUDA_VISIBLE_DEVICES=$g timeout 10800 python3 scripts/eval_stage2_q3_v1.py --attr "$a" --cond "$c" $e > "$lg" 2>&1 || true;;
    s2vl)       CUDA_VISIBLE_DEVICES=$g timeout 10800 python3 scripts/eval_stage2_vllama_v1.py --attr "$a" --cond "$c" $e > "$lg" 2>&1 || true;;
    qwen2audio) CUDA_VISIBLE_DEVICES=$g timeout 10800 python3 scripts/eval_newmodels_v1.py --model qwen2audio --attr "$a" --cond "$c" $e > "$lg" 2>&1 || true;;
    af3)        CUDA_VISIBLE_DEVICES=$g timeout 10800 bash scripts/run_af3_v1.sh --attr "$a" --cond "$c" $e > "$lg" 2>&1 || true;;
  esac; }
worker(){ local wid=$1 g=$2 job kind attr cond extra out rows need age fresh
  while :; do
    job=$(pop); [ -z "$job" ] && break
    kind=${job%%|*}; job=${job#*|}; attr=${job%%|*}; job=${job#*|}
    cond=${job%%|*}; extra=${job#*|}
    out=$(outfile "$kind" "$attr" "$cond" "$extra")
    rows=0; [ -f "$out" ] && rows=$(wc -l < "$out")
    fresh=0; if [ -f "$out" ]; then age=$(( $(date +%s) - $(stat -c %Y "$out") )); [ "$age" -lt 180 ] && fresh=1; fi
    if [ "$rows" -ge 400 ] || [ "$fresh" = 1 ]; then log "w$wid skip $kind/$attr/$cond"; continue; fi
    need=30000; [ "$kind" = nemotron ] && need=65000
    while :; do f=$(nvidia-smi -i "$g" --query-gpu=memory.free --format=csv,noheader,nounits)
      [ "$f" -gt "$need" ] && break; sleep 45; done
    run "$kind" "$attr" "$cond" "$extra" "$g" "${out%_v1.jsonl}.log"
    rows=0; [ -f "$out" ] && rows=$(wc -l < "$out")
    log "w$wid $kind/$attr/$cond $extra -> $rows"
  done; log "worker $wid done"; }
worker 1 0 & worker 2 1 & worker 3 2 & worker 4 3 &
wait
log "WAVE 3 COMPLETE"
