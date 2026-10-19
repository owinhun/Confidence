#!/bin/bash
set -u
cd /workspace/svhalluc/claude_track2
S=outputs/af3_status.txt
log(){ echo "[$(TZ=Asia/Seoul date +%H:%M)] $*" >> "$S"; }
G=${1:-2}
wait_free(){ while :; do f=$(nvidia-smi -i $G --query-gpu=memory.free --format=csv,noheader,nounits); [ "$f" -gt 30000 ] && return 0; sleep 60; done; }
run(){ local a=$1 c=$2 cl=${3:-false} tag="" n
  [ "$cl" = true ] && tag="_T"
  local out="outputs/nm_af3_${a}_${c}${tag}_v1.jsonl"
  for t in 1 2 3; do
    [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ] && { log "$a/$c$tag done"; return; }
    wait_free
    CUDA_VISIBLE_DEVICES=$G bash scripts/run_af3_v1.sh --attr "$a" --cond "$c" --claim "$cl" \
      > "outputs/nm_af3_${a}_${c}${tag}.log" 2>&1
    n=$(wc -l < "$out" 2>/dev/null || echo 0); log "$a/$c$tag try$t -> $n"
    [ "$n" -ge 400 ] && return; sleep 20
  done; log "$a/$c$tag FAILED"; }
for c in none anon_high anon_low expert_high expert_low true_high irrelevant_high irrelevant_low bare_false; do run sound "$c"; done
for c in v_vlow v_low v_mod v_high v_vhigh; do run sound "$c"; done
for c in v_vlow v_low v_mod v_high v_vhigh; do run sound "$c" true; done
for c in none anon_high anon_low true_high irrelevant_high; do run noise "$c"; done
log "AF3 ALL DONE"
