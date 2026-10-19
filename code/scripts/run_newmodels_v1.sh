#!/bin/bash
# Widening the model base. Core conditions first (the confirmatory contrast and its
# controls), then the confidence ladder, so a partial run still yields a usable cell.
set -u
cd /workspace/svhalluc/claude_track2
export HF_HOME=/data4/hf_cache HF_HUB_CACHE=/data4/hf_cache/hub
S=outputs/newmodels_status.txt
log(){ echo "[$(TZ=Asia/Seoul date +%H:%M)] $*" >> "$S"; }
wait_free(){ local g=$1
  while :; do
    f=$(nvidia-smi -i "$g" --query-gpu=memory.free --format=csv,noheader,nounits)
    [ "$f" -gt 26000 ] && return 0; sleep 60
  done; }
run(){ local g=$1 m=$2 a=$3 c=$4 cl=${5:-false}
  local tag=""; [ "$cl" = true ] && tag="_T"
  local out="outputs/nm_${m}_${a}_${c}${tag}_v1.jsonl" n
  for t in 1 2 3; do
    [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ] && { log "$m/$a/$c$tag done"; return; }
    wait_free "$g"
    CUDA_VISIBLE_DEVICES=$g python3 scripts/eval_newmodels_v1.py --model "$m" --attr "$a" \
      --cond "$c" --claim "$cl" > "outputs/nm_${m}_${a}_${c}${tag}.log" 2>&1
    n=$(wc -l < "$out" 2>/dev/null || echo 0); log "$m/$a/$c$tag try$t -> $n"
    [ "$n" -ge 400 ] && return; sleep 20
  done; log "$m/$a/$c$tag FAILED"; }

M=${1:-qwen2audio}; G=${2:-0}
for c in none anon_high anon_low expert_high expert_low true_high irrelevant_high irrelevant_low bare_false; do
  run "$G" "$M" sound "$c"
done
for c in v_vlow v_low v_mod v_high v_vhigh; do run "$G" "$M" sound "$c"; done
for c in v_vlow v_low v_mod v_high v_vhigh; do run "$G" "$M" sound "$c" true; done
for c in none anon_high anon_low true_high irrelevant_high; do run "$G" "$M" noise "$c"; done
log "$M ALL DONE"
