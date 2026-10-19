#!/bin/bash
# Decisive control for the retracted "signed evidence" mechanism, plus the numeric
# true-curves needed before any claim that the neutral point is model-specific.
set -u
cd /workspace/svhalluc/claude_track2
S=outputs/irrel_status.txt
log(){ echo "[$(TZ=Asia/Seoul date +%H:%M)] $*" >> "$S"; }
wait_free(){ local need=$1; shift; local g tot
  while :; do tot=0
    for g in "$@"; do tot=$((tot+$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits))); done
    [ "$tot" -gt "$need" ] && return 0; sleep 60
  done; }
s2(){ local gpus="$1" scr=$2 pfx=$3 c=$4 need=$5 out n
  out="outputs/${pfx}_sound_${c}_v1.jsonl"
  for t in 1 2 3; do
    [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ] && { log "$pfx/$c done"; return; }
    wait_free "$need" $gpus
    CUDA_VISIBLE_DEVICES=$(echo $gpus|tr ' ' ,) python3 "$scr" --attr sound --cond "$c" \
      > "outputs/${pfx}_sound_${c}.log" 2>&1
    n=$(wc -l < "$out" 2>/dev/null||echo 0); log "$pfx/$c try$t -> $n"; [ "$n" -ge 400 ] && return; sleep 20
  done; log "$pfx/$c FAILED"; }
cs(){ local gpus="$1" m=$2 c=$3 need=$4 out n
  out="outputs/cs_${m}_sound_${c}_T_v1.jsonl"
  for t in 1 2 3; do
    [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ] && { log "cs $m/$c done"; return; }
    wait_free "$need" $gpus
    CUDA_VISIBLE_DEVICES=$(echo $gpus|tr ' ' ,) python3 scripts/eval_confscale_v1.py \
      --model "$m" --attr sound --cond "$c" --claim true > "outputs/cs_${m}_sound_${c}_T.log" 2>&1
    n=$(wc -l < "$out" 2>/dev/null||echo 0); log "cs $m/$c try$t -> $n"; [ "$n" -ge 400 ] && return; sleep 20
  done; log "cs $m/$c FAILED"; }

# lane A (gpu0): Qwen2.5 -- control first, then its remaining numeric true-curve is done
( s2 0 scripts/eval_stage2_factorial_v1.py s2 irrelevant_low 27000
  for c in n_015 n_035 n_060 n_085 n_095; do cs 0 vllama $c 27000; done
  log "LANE-A DONE" ) &
# lane B (gpu1): VideoLLaMA control
( s2 1 scripts/eval_stage2_vllama_v1.py s2vl irrelevant_low 27000
  log "LANE-B DONE" ) &
# lane C (gpus 2+3): Qwen3 control, then its numeric true-curve
( s2 "2 3" scripts/eval_stage2_q3_v1.py s2q3 irrelevant_low 78000
  for c in n_015 n_035 n_060 n_085 n_095; do cs "2 3" qwen3 $c 78000; done
  log "LANE-C DONE" ) &
wait
log "IRREL-LOW + NUMERIC-TRUE ALL DONE"
