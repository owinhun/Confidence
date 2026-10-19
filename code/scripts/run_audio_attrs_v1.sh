#!/bin/bash
# Stage 2 on three NEW auditory attributes (noise / lowpass / reverb), all three models.
#
# Purpose: the confirmatory design currently rests on a single auditory attribute. These
# three are pure signal-processing manipulations with exact by-construction labels, so if
# the confidence effect holds on them the result is about audition, not about semantic
# event labels.
#
# Scheduling: a shared job queue with workers pinned to GPU slots. Other tenants hold
# 20-40 GiB per card, so each worker waits on FREE memory for its own model rather than
# assuming an empty card. The 30B model is sharded over two cards for the same reason.
set -u
CT2=/workspace/svhalluc/claude_track2; cd "$CT2"
S=/root/.claude/jobs/65f6dcfb/tmp/audio_status.txt
Q=/root/.claude/jobs/65f6dcfb/tmp/audio_queue
LOCK=/root/.claude/jobs/65f6dcfb/tmp/audio_queue.lock
log(){ echo "[$(date -u +%H:%M)] $*" >> "$S"; }

ATTRS="noise lowpass reverb"
CONDS="none anon_high anon_low true_high irrelevant_high bare_false"

SCRIPT_FOR(){ case "$1" in
  qwen25) echo scripts/eval_stage2_factorial_v1.py;;
  qwen3)  echo scripts/eval_stage2_q3_v1.py;;
  vllama) echo scripts/eval_stage2_vllama_v1.py;;
esac; }
PREFIX_FOR(){ case "$1" in qwen25) echo s2;; qwen3) echo s2q3;; vllama) echo s2vl;; esac; }
NEED_FOR(){ case "$1" in qwen3) echo 78000;; *) echo 27000;; esac; }

build_queue(){ : > "$Q"
  for m in $1; do for a in $ATTRS; do for c in $CONDS; do
    echo "$m $a $c" >> "$Q"
  done; done; done; }

pop(){ local line
  exec 9>"$LOCK"; flock 9
  line=$(head -n 1 "$Q"); [ -n "$line" ] && sed -i '1d' "$Q"
  flock -u 9; exec 9>&-
  echo "$line"; }

wait_mem(){ local need=$1; shift; local gpus="$*" g f tot
  while :; do
    tot=0
    for g in $gpus; do
      f=$(nvidia-smi -i "$g" --query-gpu=memory.free --format=csv,noheader,nounits)
      tot=$((tot + f))
    done
    [ "$tot" -gt "$need" ] && return 0
    sleep 60
  done; }

worker(){ local wid=$1; shift; local gpus="$*"
  local cvd; cvd=$(echo "$gpus" | tr ' ' ',')
  while :; do
    local job; job=$(pop); [ -z "$job" ] && break
    set -- $job; local m=$1 a=$2 c=$3
    local pfx; pfx=$(PREFIX_FOR "$m")
    local out="outputs/${pfx}_${a}_${c}_v1.jsonl"
    local need; need=$(NEED_FOR "$m")
    local n=0
    for try in 1 2 3; do
      if [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ]; then n=400; break; fi
      wait_mem "$need" $gpus
      CUDA_VISIBLE_DEVICES=$cvd python3 "$(SCRIPT_FOR "$m")" --attr "$a" --cond "$c" \
        > "outputs/${pfx}_${a}_${c}.log" 2>&1
      n=$(wc -l < "$out" 2>/dev/null || echo 0)
      [ "$n" -ge 400 ] && break
      sleep 20
    done
    log "w$wid $m/$a/$c -> $n"
  done
  log "worker $wid done"; }

# ---- phase 1: the two 7B models, four workers, two per card -------------------
build_queue "qwen25 vllama"
worker 1 0 & worker 2 0 & worker 3 1 & worker 4 1 &
worker 5 2 & worker 6 2 & worker 7 3 & worker 8 3 &
wait
log "PHASE1 (7B models) DONE"

# ---- phase 2 (amended 2026-08-10): eligibility screen only ------------------
# Every completed low-level cell failed the perceptual-eligibility gate, so the
# confidence-condition runs are excluded from analysis a priori. Only the three
# 'none' runs are needed to complete the eligibility table for Qwen3.
: > "$Q"
for a in $ATTRS; do echo "qwen3 $a none" >> "$Q"; done
worker 9 0 1 & worker 10 2 3 &
wait
log "AUDIO-ATTRS ALL DONE (phase2 = eligibility only)"
