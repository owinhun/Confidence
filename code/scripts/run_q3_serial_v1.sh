#!/bin/bash
# Remaining Qwen3-30B work, strictly serial (parallel lanes were OOM-racing each other):
#   1) the one ELIGIBLE low-level cell (noise) -- its five confidence conditions
#   2) the qwen3 aux-task A0 screen closure
set -u
cd /workspace/svhalluc/claude_track2
S=outputs/q3serial_status.txt
log(){ echo "[$(TZ=Asia/Seoul date +%H:%M)] $*" >> "$S"; }
pick_pair(){ local g h f fh
  while :; do
    for g in 0 1 2 3; do for h in 0 1 2 3; do
      [ "$g" -ge "$h" ] && continue
      f=$(nvidia-smi -i "$g" --query-gpu=memory.free --format=csv,noheader,nounits)
      fh=$(nvidia-smi -i "$h" --query-gpu=memory.free --format=csv,noheader,nounits)
      [ $((f+fh)) -gt 84000 ] && { echo "$g,$h"; return 0; }
    done; done
    sleep 120
  done; }
run_s2(){ local c=$1 out="outputs/s2q3_noise_$1_v1.jsonl" n P
  for try in 1 2 3; do
    [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ] && { log "s2 noise/$c done"; return; }
    P=$(pick_pair)
    CUDA_VISIBLE_DEVICES=$P python3 scripts/eval_stage2_q3_v1.py --attr noise --cond "$c" \
      > "outputs/s2q3_noise_${c}.log" 2>&1
    n=$(wc -l < "$out" 2>/dev/null || echo 0); log "s2 noise/$c try$try [$P] -> $n"
    [ "$n" -ge 400 ] && return; sleep 30
  done; log "s2 noise/$c FAILED"; }
run_cs(){ local a=$1 out="outputs/cs_qwen3_$1_none_v1.jsonl" n P
  for try in 1 2 3; do
    [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ] && { log "cs $a done"; return; }
    P=$(pick_pair)
    CUDA_VISIBLE_DEVICES=$P python3 scripts/eval_confscale_v1.py --model qwen3 \
      --attr "$a" --cond none > "outputs/cs_qwen3_${a}_none.log" 2>&1
    n=$(wc -l < "$out" 2>/dev/null || echo 0); log "cs $a try$try [$P] -> $n"
    [ "$n" -ge 400 ] && return; sleep 30
  done; log "cs $a FAILED"; }
for c in anon_high anon_low true_high irrelevant_high bare_false; do run_s2 "$c"; done
for a in aux_noise aux_lowpass aux_reverb; do run_cs "$a"; done
log "Q3-SERIAL ALL DONE"
