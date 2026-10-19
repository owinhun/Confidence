#!/bin/bash
# Confidence dose-response: the ICASSP contribution -- a characterised control knob for the
# detector->LLM cascade, not just a diagnosis. Sound event only (the audio-side attribute that
# passes every Stage-2 control on all three models).
set -u
CT2=/workspace/svhalluc/claude_track2; cd "$CT2"
S=/root/.claude/jobs/65f6dcfb/tmp/cs_status.txt
log(){ echo "[$(date -u +%H:%M)] $*" | tee -a "$S"; }

# The cards are shared with other tenants, so wait on FREE memory against what this particular
# model actually needs, rather than on the card being empty.
need_for(){ case "$1" in qwen3) echo 78000;; *) echo 26000;; esac; }

wait_free(){ local g=$1 need=$2 f
  while :; do
    f=$(nvidia-smi -i "$g" --query-gpu=memory.free --format=csv,noheader,nounits)
    [ "$f" -gt "$need" ] && return 0
    sleep 90
  done; }

pick_gpu(){ local need=$1 g h f fh     # blocks until enough room exists on 0..3
  while :; do
    for g in 0 1 2 3; do
      f=$(nvidia-smi -i "$g" --query-gpu=memory.free --format=csv,noheader,nounits)
      [ "$f" -gt "$need" ] && { echo "$g"; return 0; }
    done
    # No single card fits. The 30B model is loaded with device_map="auto", so it can be
    # sharded over two cards -- necessary here because other tenants hold 30-40 GiB on
    # every card and a whole 80 GiB card never frees up.
    if [ "$need" -gt 40000 ]; then
      for g in 0 1 2 3; do
        for h in 0 1 2 3; do
          [ "$g" -ge "$h" ] && continue
          f=$(nvidia-smi -i "$g" --query-gpu=memory.free --format=csv,noheader,nounits)
          fh=$(nvidia-smi -i "$h" --query-gpu=memory.free --format=csv,noheader,nounits)
          [ $((f + fh)) -gt $((need + 6000)) ] && { echo "$g,$h"; return 0; }
        done
      done
    fi
    sleep 120
  done; }

run(){ local m=$1 c=$2 need g out try n
  need=$(need_for "$m")
  out="outputs/cs_${m}_sound_${c}_v1.jsonl"
  for try in 1 2 3; do
    [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ] && { log "cs $m/$c complete"; return 0; }
    g=$(pick_gpu "$need")
    CUDA_VISIBLE_DEVICES=$g python3 scripts/eval_confscale_v1.py --model "$m" \
      --attr sound --cond "$c" > "outputs/cs_${m}_sound_${c}.log" 2>&1
    n=$(wc -l < "$out" 2>/dev/null || echo 0)
    log "cs $m/$c gpu$g try$try -> $n"
    [ "$n" -ge 400 ] && return 0
    sleep 20
  done
  log "cs $m/$c FAILED"; }

V="v_vlow v_low v_mod v_high v_vhigh"
N="n_015 n_035 n_060 n_085 n_095"

( for c in $V $N; do run qwen25 "$c"; done; log "CS-QWEN25 DONE" ) &
( for c in $V $N; do run vllama "$c"; done; log "CS-VLLAMA DONE" ) &
( for c in $V $N; do run qwen3  "$c"; done; log "CS-QWEN3 DONE" ) &
wait
log "CONFSCALE ALL DONE"
