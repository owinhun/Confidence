#!/bin/bash
# Follow-on after the H5 demotion: finish the eligibility table, then screen the
# auxiliary-evidence task (A0 on degraded+clean clips, sound-event question).
set -u
CT2=/workspace/svhalluc/claude_track2; cd "$CT2"
S=/root/.claude/jobs/65f6dcfb/tmp/aux_status.txt
log(){ echo "[$(date -u +%H:%M)] $*" | tee -a "$S"; }
wait_mem(){ local need=$1; shift; local g tot
  while :; do tot=0
    for g in "$@"; do tot=$((tot + $(nvidia-smi -i "$g" --query-gpu=memory.free --format=csv,noheader,nounits))); done
    [ "$tot" -gt "$need" ] && return 0; sleep 60
  done; }
s2run(){ local gpus="$1" model_script=$2 pfx=$3 a=$4 c=$5 need=$6
  local out="outputs/${pfx}_${a}_${c}_v1.jsonl" cvd n
  cvd=$(echo "$gpus" | tr ' ' ',')
  for try in 1 2 3; do
    [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ] && { log "$pfx/$a/$c done"; return 0; }
    wait_mem "$need" $gpus
    CUDA_VISIBLE_DEVICES=$cvd python3 "$model_script" --attr "$a" --cond "$c" \
      > "outputs/${pfx}_${a}_${c}.log" 2>&1
    n=$(wc -l < "$out" 2>/dev/null || echo 0); log "$pfx/$a/$c try$try -> $n"
    [ "$n" -ge 400 ] && return 0; sleep 20
  done; log "$pfx/$a/$c FAILED"; }
csrun(){ local gpus="$1" m=$2 a=$3 c=$4 need=$5
  local out="outputs/cs_${m}_${a}_${c}_v1.jsonl" cvd n
  cvd=$(echo "$gpus" | tr ' ' ',')
  for try in 1 2 3; do
    [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ] && { log "cs $m/$a/$c done"; return 0; }
    wait_mem "$need" $gpus
    CUDA_VISIBLE_DEVICES=$cvd python3 scripts/eval_confscale_v1.py --model "$m" \
      --attr "$a" --cond "$c" > "outputs/cs_${m}_${a}_${c}.log" 2>&1
    n=$(wc -l < "$out" 2>/dev/null || echo 0); log "cs $m/$a/$c try$try -> $n"
    [ "$n" -ge 400 ] && return 0; sleep 20
  done; log "cs $m/$a/$c FAILED"; }

# lane 1 (gpu0): finish the vllama eligibility 'none' rows, then vllama aux A0
( s2run 0 scripts/eval_stage2_vllama_v1.py s2vl lowpass none 27000
  s2run 0 scripts/eval_stage2_vllama_v1.py s2vl reverb none 27000
  for a in aux_noise aux_lowpass aux_reverb; do csrun 0 vllama $a none 27000; done
  log "LANE1 DONE" ) &
# lane 2 (gpu1): qwen25 aux A0
( for a in aux_noise aux_lowpass aux_reverb; do csrun 1 qwen25 $a none 27000; done
  log "LANE2 DONE" ) &
# lane 3 (gpus 2+3): qwen3 eligibility, then qwen3 aux A0
( for a in noise lowpass reverb; do s2run "2 3" scripts/eval_stage2_q3_v1.py s2q3 $a none 78000; done
  for a in aux_noise aux_lowpass aux_reverb; do csrun "2 3" qwen3 $a none 78000; done
  log "LANE3 DONE" ) &
wait
log "AUX-SCREEN ALL DONE"
