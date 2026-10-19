#!/bin/bash
# The identification battery: everything needed to say what the CONFIDENCE FIELD does, as
# opposed to what the claim attached to it does.
#
# Three gaps close here.
#
#  (1) priority_true_{high,low} -- the non-epistemic field carrying a TRUE claim. Paired with
#      the existing priority_{high,low} it gives the claim-aligned contrast for a field that
#      declares no confidence. That is the number the declared-confidence contrast has to beat
#      before it can be called an effect of declared confidence rather than of any filled field.
#
#  (2) the confidence ladder for VideoLLaMA on COLOUR. The ladder only ever ran on sound, and
#      VideoLLaMA's sound cell was withdrawn when the option swap put its unaided accuracy at
#      chance -- so the figure was plotting a cell the text retracts. Colour is the cell that
#      survived the swap, so the ladder has to move there.
#
#  (3) the one-step contrast conditions VideoLLaMA never had (null_block, the epistemic
#      synonyms, the Priority field) plus a truncated AF3 run, so the decomposition table has
#      every model in it rather than the three that happened to be implemented.
#
# Qwen3-Omni is 30B and needs a GPU pair, so its jobs run last and serially rather than
# competing with the four single-card workers.
set -u
cd /workspace/svhalluc/claude_track2
S=outputs/identify_status.txt
Q=/root/.claude/jobs/65f6dcfb/tmp/identify_queue
QP=/root/.claude/jobs/65f6dcfb/tmp/identify_pair_queue
LK=/root/.claude/jobs/65f6dcfb/tmp/identify.lock
K(){ python3 -c "import datetime;print((datetime.datetime.utcnow()+datetime.timedelta(hours=9)).strftime('%H:%M'))"; }
log(){ echo "[KST $(K)] $*" >> "$S"; }

# do not fight the graded screen for cards
while [ -s /root/.claude/jobs/65f6dcfb/tmp/graded_queue ]; do sleep 60; done
while pgrep -f "eval_stage2_factorial_v1.py --attr grad\|eval_newmodels_v1.py --model qwen2audio --attr grad" > /dev/null; do sleep 60; done
log "graded screen clear; identification battery starting"

: > "$Q"; : > "$QP"
# --- (1) non-epistemic field with a true claim, on the sound cell ---
for c in priority_true_high priority_true_low; do
  echo "s2|$c|sound"            >> "$Q"
  echo "qwen2audio|$c|sound"    >> "$Q"
  echo "af3|$c|sound"           >> "$Q"
  echo "q3|$c|sound"            >> "$QP"
done
# --- resume the AF3 run that stopped at 193/400 ---
echo "af3|priority_high|sound" >> "$Q"
# --- (2) the ladder on VideoLLaMA's surviving cell ---
for c in v_vlow v_low v_mod v_high v_vhigh; do
  echo "vl_lad|$c|colour"   >> "$Q"
  echo "vl_ladT|$c|colour"  >> "$Q"
done
# --- (3) VideoLLaMA's missing one-step contrasts ---
for c in null_block reliab_high reliab_low certain_high certain_low \
         priority_high priority_low priority_true_high priority_true_low; do
  echo "vl|$c|colour" >> "$Q"
done
log "queued $(wc -l < "$Q") single-card + $(wc -l < "$QP") paired"

outfile(){ # kind cond attr -> jsonl path
  case "$1" in
    s2)         echo "outputs/s2_$3_$2_v1.jsonl" ;;
    q3)         echo "outputs/s2q3_$3_$2_v1.jsonl" ;;
    qwen2audio) echo "outputs/nm_qwen2audio_$3_$2_v1.jsonl" ;;
    af3)        echo "outputs/nm_af3_$3_$2_v1.jsonl" ;;
    vl)         echo "outputs/s2vl_$3_$2_v1.jsonl" ;;
    vl_lad)     echo "outputs/cs_vllama_$3_$2_v1.jsonl" ;;
    vl_ladT)    echo "outputs/cs_vllama_$3_$2_T_v1.jsonl" ;;
  esac; }

launch(){ # kind cond attr gpuspec logsuffix
  local kind=$1 cond=$2 attr=$3 g=$4 lg=$5
  case "$kind" in
    s2)         CUDA_VISIBLE_DEVICES=$g python3 scripts/eval_stage2_factorial_v1.py --attr "$attr" --cond "$cond" > "$lg" 2>&1 ;;
    q3)         CUDA_VISIBLE_DEVICES=$g python3 scripts/eval_stage2_q3_v1.py --attr "$attr" --cond "$cond" > "$lg" 2>&1 ;;
    qwen2audio) CUDA_VISIBLE_DEVICES=$g python3 scripts/eval_newmodels_v1.py --model qwen2audio --attr "$attr" --cond "$cond" > "$lg" 2>&1 ;;
    af3)        CUDA_VISIBLE_DEVICES=$g bash scripts/run_af3_v1.sh --attr "$attr" --cond "$cond" > "$lg" 2>&1 ;;
    vl)         CUDA_VISIBLE_DEVICES=$g python3 scripts/eval_stage2_vllama_v1.py --attr "$attr" --cond "$cond" > "$lg" 2>&1 ;;
    vl_lad)     CUDA_VISIBLE_DEVICES=$g python3 scripts/eval_confscale_v1.py --model vllama --attr "$attr" --cond "$cond" --claim false > "$lg" 2>&1 ;;
    vl_ladT)    CUDA_VISIBLE_DEVICES=$g python3 scripts/eval_confscale_v1.py --model vllama --attr "$attr" --cond "$cond" --claim true  > "$lg" 2>&1 ;;
  esac; }

pop(){ exec 9>"$LK"; flock 9; local l; l=$(head -1 "$1"); [ -n "$l" ] && sed -i '1d' "$1"; flock -u 9; exec 9>&-; echo "$l"; }

worker(){ local wid=$1 g=$2
  while :; do
    local job; job=$(pop "$Q"); [ -z "$job" ] && break
    IFS='|' read -r kind cond attr <<< "$job"
    local out; out=$(outfile "$kind" "$cond" "$attr")
    local lg="${out%_v1.jsonl}.log" n=0
    for t in 1 2 3; do
      [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ] && { n=400; break; }
      while :; do f=$(nvidia-smi -i "$g" --query-gpu=memory.free --format=csv,noheader,nounits); [ "$f" -gt 28000 ] && break; sleep 60; done
      launch "$kind" "$cond" "$attr" "$g" "$lg"
      n=$(wc -l < "$out" 2>/dev/null || echo 0); [ "$n" -ge 400 ] && break; sleep 20
    done
    log "w$wid $kind/$attr/$cond -> $n"
  done; log "worker $wid done"; }

# GPUs 0 and 1 have ~55 GB free and a 7B job takes ~25 GB, so they carry two workers
# each; 2 and 3 are tighter and carry one. The in-worker free-memory guard (>28 GB)
# still gates every launch, so a card that fills up simply stalls its second worker.
worker 1 0 & worker 2 0 & worker 3 1 & worker 4 1 & worker 5 2 & worker 6 3 &
wait
log "single-card phase done; Qwen3 pair phase starting"

while :; do
  job=$(pop "$QP"); [ -z "$job" ] && break
  IFS='|' read -r kind cond attr <<< "$job"
  out=$(outfile "$kind" "$cond" "$attr"); lg="${out%_v1.jsonl}.log"; n=0
  for t in 1 2 3; do
    [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ] && { n=400; break; }
    while :; do
      a=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits)
      b=$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits)
      [ "$a" -gt 40000 ] && [ "$b" -gt 40000 ] && break; sleep 60
    done
    launch "$kind" "$cond" "$attr" "0,1" "$lg"
    n=$(wc -l < "$out" 2>/dev/null || echo 0); [ "$n" -ge 400 ] && break; sleep 20
  done
  log "pair $kind/$attr/$cond -> $n"
done
log "IDENTIFICATION BATTERY DONE"
