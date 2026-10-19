#!/bin/bash
# Next-wave runner on a clean namespace.
# - never writes to the original outputs
# - starts with a preloaded queue and independent lock
# - uses GPUs 0/1/2/3 in lockstep
set -u
cd /workspace/svhalluc/claude_track2
mkdir -p outputs/run_next
S=outputs/run_next/run_next_status.txt
Q=/root/.claude/jobs/65f6dcfb/tmp/run_next_wave2_queue
LK=/root/.claude/jobs/65f6dcfb/tmp/run_next_wave2.lock
K(){ python3 -c "import datetime;print((datetime.datetime.utcnow()+datetime.timedelta(hours=9)).strftime('%m-%d %H:%M'))"; }
log(){ echo "[KST $(K)] $*" >> "$S"; }

pop(){
  exec 9>"$LK"; flock 9
  local l; l=$(head -1 "$Q" 2>/dev/null || true)
  [ -n "$l" ] && sed -i '1d' "$Q"
  flock -u 9; exec 9>&-
  printf '%s' "$l"
}

outfile(){
  local k=$1 a=$2 c=$3 e=$4 sfx=""
  case "$e" in *swap*) sfx="_swap";; esac
  case "$k" in
    nemotron)   printf 'outputs/run_next/nm_nemotron_%s_%s%s_v1.jsonl' "$a" "$c" "$sfx";;
    s2)         printf 'outputs/run_next/s2_%s_%s%s_v1.jsonl' "$a" "$c" "$sfx";;
    s2q3)       printf 'outputs/run_next/s2q3_%s_%s%s_v1.jsonl' "$a" "$c" "$sfx";;
    s2vl)       printf 'outputs/run_next/s2vl_%s_%s%s_v1.jsonl' "$a" "$c" "$sfx";;
    qwen2audio) printf 'outputs/run_next/nm_qwen2audio_%s_%s%s_v1.jsonl' "$a" "$c" "$sfx";;
    af3)        printf 'outputs/run_next/nm_af3_%s_%s%s_v1.jsonl' "$a" "$c" "$sfx";;
    llava_next) printf 'outputs/run_next/gm_llava_next_video_%s_%s%s_v1.jsonl' "$a" "$c" "$sfx";;
  esac
}

run(){
  local k=$1 a=$2 c=$3 e=$4 g=$5 lg=$6
  case "$k" in
    nemotron)   CUDA_VISIBLE_DEVICES=$g timeout 10800 bash scripts/run_nemotron_v1.sh --attr "$a" --cond "$c" $e > "$lg" 2>&1 || true;;
    s2)         CUDA_VISIBLE_DEVICES=$g timeout 10800 python3 scripts/eval_stage2_factorial_v1.py --attr "$a" --cond "$c" $e > "$lg" 2>&1 || true;;
    s2q3)       CUDA_VISIBLE_DEVICES=$g timeout 10800 python3 scripts/eval_stage2_q3_v1.py --attr "$a" --cond "$c" $e > "$lg" 2>&1 || true;;
    s2vl)       CUDA_VISIBLE_DEVICES=$g timeout 10800 python3 scripts/eval_stage2_vllama_v1.py --attr "$a" --cond "$c" $e > "$lg" 2>&1 || true;;
    qwen2audio) CUDA_VISIBLE_DEVICES=$g timeout 10800 python3 scripts/eval_newmodels_v1.py --model qwen2audio --attr "$a" --cond "$c" $e > "$lg" 2>&1 || true;;
    af3)        CUDA_VISIBLE_DEVICES=$g timeout 10800 bash scripts/run_af3_v1.sh --attr "$a" --cond "$c" $e > "$lg" 2>&1 || true;;
    llava_next) CUDA_VISIBLE_DEVICES=$g timeout 10800 python3 scripts/eval_generic_v1.py --model llava_next_video --attr "$a" --cond "$c" $e > "$lg" 2>&1 || true;;
  esac
}

build_queue(){
  : > "$Q"
  for spec in "s2 sound" "s2q3 sound" "s2vl colour" "qwen2audio sound" "af3 sound"; do
    set -- $spec
    echo "$1|$2|true_low|"              >> "$Q"
    echo "$1|$2|true_low|--swap_options" >> "$Q"
  done
  for a in sound colour; do
    echo "nemotron|$a|true_low|--swap_options" >> "$Q"
    echo "nemotron|$a|anon_high|--swap_options" >> "$Q"
    echo "nemotron|$a|anon_low|--swap_options" >> "$Q"
  done
  echo "llava_next|colour|none|" >> "$Q"
  echo "llava_next|colour|none|--swap_options" >> "$Q"
  echo "qwen2audio|grad_noise_g4|anon_high|" >> "$Q"
  echo "qwen2audio|grad_noise_g4|anon_low|" >> "$Q"
  log "queued $(wc -l < "$Q") conditions"
}

worker(){
  local wid=$1 g=$2 job kind attr cond extra out need rows age fresh
  while :; do
    job=$(pop)
    [ -z "$job" ] && break
    kind=${job%%|*}; job=${job#*|}
    attr=${job%%|*}; job=${job#*|}
    cond=${job%%|*}
    extra=${job#*|}
    out=$(outfile "$kind" "$attr" "$cond" "$extra")
    if [ -z "$out" ]; then log "w$wid unknown harness '$kind'"; continue; fi
    rows=0; [ -f "$out" ] && rows=$(wc -l < "$out")
    fresh=0
    if [ -f "$out" ]; then
      age=$(( $(date +%s) - $(stat -c %Y "$out") ))
      [ "$age" -lt 180 ] && fresh=1
    fi
    if [ "$rows" -ge 400 ] || [ "$fresh" = 1 ]; then
      log "w$wid skip $kind/$attr/$cond (rows=$rows fresh=$fresh)"; continue
    fi
    need=30000; [ "$kind" = nemotron ] && need=65000
    while :; do
      f=$(nvidia-smi -i "$g" --query-gpu=memory.free --format=csv,noheader,nounits)
      [ "$f" -gt "$need" ] && break
      sleep 45
    done
    run "$kind" "$attr" "$cond" "$extra" "$g" "${out%_v1.jsonl}.log"
    rows=0; [ -f "$out" ] && rows=$(wc -l < "$out")
    log "w$wid $kind/$attr/$cond ${extra} -> $rows"
  done
  log "worker $wid done"
}

build_queue
log "run_next_wave2 starting; $(wc -l < "$Q") conditions queued"
worker 1 0 & worker 2 1 & worker 3 2 & worker 4 3 &
wait
log "BACKLOG COMPLETE"
