#!/bin/bash
# Next wave. Three jobs, chosen so the paper stops being internally inconsistent:
#
#  A. true_low, both orders, on every cell. Table 1 is now order-averaged but the beta ratio is
#     not, and the ratio is the paper's identified result -- leaving it single-order while the
#     table beside it is averaged is exactly the asymmetry a reviewer picks at. The five Qwen
#     cells never had true_low at all (the ratio borrowed the ladder's v_low_T), so it is new
#     data, not just a swapped half.
#  B. Nemotron's anon_high/anon_low swapped halves, which the last queue omitted.
#  C. LLaVA-NeXT-Video from the converted llava-hf checkpoint -- the earlier failure was the
#     lmms-lab layout having no image-processor config, not the model.
#  D. Qwen2-Audio's two missing graded-noise contrast conditions.
set -u
cd /workspace/svhalluc/claude_track2
S=outputs/wave2_status.txt
Q=/root/.claude/jobs/65f6dcfb/tmp/wave2_queue
LK=/root/.claude/jobs/65f6dcfb/tmp/wave2.lock
K(){ python3 -c "import datetime;print((datetime.datetime.utcnow()+datetime.timedelta(hours=9)).strftime('%m-%d %H:%M'))"; }
log(){ echo "[KST $(K)] $*" >> "$S"; }

: > "$Q"
for spec in "s2 sound" "s2q3 sound" "s2vl colour" "qwen2audio sound" "af3 sound"; do
  set -- $spec
  echo "$1|$2|true_low|"                >> "$Q"
  echo "$1|$2|true_low|--swap_options"  >> "$Q"
done
for a in sound colour; do
  echo "nemotron|$a|true_low|--swap_options" >> "$Q"
  echo "nemotron|$a|anon_high|--swap_options" >> "$Q"
  echo "nemotron|$a|anon_low|--swap_options"  >> "$Q"
done
echo "llava_next|colour|none|"               >> "$Q"
echo "llava_next|colour|none|--swap_options" >> "$Q"
echo "qwen2audio|grad_noise_g4|anon_high|"   >> "$Q"
echo "qwen2audio|grad_noise_g4|anon_low|"    >> "$Q"
log "queued $(wc -l < "$Q") conditions"

pop(){ exec 9>"$LK"; flock 9; local l; l=$(head -1 "$Q" 2>/dev/null || true)
  [ -n "$l" ] && sed -i '1d' "$Q"; flock -u 9; exec 9>&-; printf '%s' "$l"; }

outfile(){ local k=$1 a=$2 c=$3 e=$4 sfx=""
  case "$e" in *swap*) sfx="_swap";; esac
  case "$k" in
    nemotron)   printf 'outputs/nm_nemotron_%s_%s%s_v1.jsonl' "$a" "$c" "$sfx";;
    s2)         printf 'outputs/s2_%s_%s%s_v1.jsonl' "$a" "$c" "$sfx";;
    s2q3)       printf 'outputs/s2q3_%s_%s%s_v1.jsonl' "$a" "$c" "$sfx";;
    s2vl)       printf 'outputs/s2vl_%s_%s%s_v1.jsonl' "$a" "$c" "$sfx";;
    qwen2audio) printf 'outputs/nm_qwen2audio_%s_%s%s_v1.jsonl' "$a" "$c" "$sfx";;
    af3)        printf 'outputs/nm_af3_%s_%s%s_v1.jsonl' "$a" "$c" "$sfx";;
    llava_next) printf 'outputs/gm_llava_next_video_%s_%s%s_v1.jsonl' "$a" "$c" "$sfx";;
  esac; }

run(){ local k=$1 a=$2 c=$3 e=$4 g=$5 lg=$6
  case "$k" in
    nemotron)   CUDA_VISIBLE_DEVICES=$g timeout 10800 bash scripts/run_nemotron_v1.sh --attr "$a" --cond "$c" $e > "$lg" 2>&1 || true;;
    s2)         CUDA_VISIBLE_DEVICES=$g timeout 10800 python3 scripts/eval_stage2_factorial_v1.py --attr "$a" --cond "$c" $e > "$lg" 2>&1 || true;;
    s2q3)       CUDA_VISIBLE_DEVICES=$g timeout 10800 python3 scripts/eval_stage2_q3_v1.py --attr "$a" --cond "$c" $e > "$lg" 2>&1 || true;;
    s2vl)       CUDA_VISIBLE_DEVICES=$g timeout 10800 python3 scripts/eval_stage2_vllama_v1.py --attr "$a" --cond "$c" $e > "$lg" 2>&1 || true;;
    qwen2audio) CUDA_VISIBLE_DEVICES=$g timeout 10800 python3 scripts/eval_newmodels_v1.py --model qwen2audio --attr "$a" --cond "$c" $e > "$lg" 2>&1 || true;;
    af3)        CUDA_VISIBLE_DEVICES=$g timeout 10800 bash scripts/run_af3_v1.sh --attr "$a" --cond "$c" $e > "$lg" 2>&1 || true;;
    llava_next) CUDA_VISIBLE_DEVICES=$g timeout 10800 python3 scripts/eval_generic_v1.py --model llava_next_video --attr "$a" --cond "$c" $e > "$lg" 2>&1 || true;;
  esac; }

worker(){ local wid=$1 g=$2 job kind attr cond extra out rows need age fresh
  while :; do
    job=$(pop); [ -z "$job" ] && break
    kind=${job%%|*}; job=${job#*|}; attr=${job%%|*}; job=${job#*|}
    cond=${job%%|*}; extra=${job#*|}
    out=$(outfile "$kind" "$attr" "$cond" "$extra")
    rows=0; [ -f "$out" ] && rows=$(wc -l < "$out")
    fresh=0
    if [ -f "$out" ]; then age=$(( $(date +%s) - $(stat -c %Y "$out") )); [ "$age" -lt 180 ] && fresh=1; fi
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
log "WAVE 2 COMPLETE"
