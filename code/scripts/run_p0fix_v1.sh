#!/bin/bash
# P0: make the decomposition actually one-step, and finish the option-order audit.
#
# The published Table 1 called its middle column "+ claim", but the step from null_block to
# priority_c moved four things at once: Source (--- -> anon), Claim ("" -> "F"), the field name
# (Confidence -> Priority) and the field value (--- -> c). srconly_c supplies the missing rung:
# Source and a filled non-epistemic field, no claim. srconly_c -> priority_c is then claim
# insertion alone, and priority_c -> anon_c is field semantics alone, as already claimed.
#
# Also runs the swapped-option pass on the newly eligible acoustic cell, which is the only
# primary cell without one.
set -u
cd /workspace/svhalluc/claude_track2
S=outputs/p0fix_status.txt
Q=/root/.claude/jobs/65f6dcfb/tmp/p0fix_queue
QP=/root/.claude/jobs/65f6dcfb/tmp/p0fix_pair_queue
LK=/root/.claude/jobs/65f6dcfb/tmp/p0fix.lock
K(){ python3 -c "import datetime;print((datetime.datetime.utcnow()+datetime.timedelta(hours=9)).strftime('%H:%M'))"; }
log(){ echo "[KST $(K)] $*" >> "$S"; }
: > "$Q"; : > "$QP"
for c in srconly_high srconly_low; do
  echo "s2|$c|sound|"          >> "$Q"
  echo "qwen2audio|$c|sound|"  >> "$Q"
  echo "af3|$c|sound|"         >> "$Q"
  echo "vl|$c|colour|"         >> "$Q"
  echo "q3|$c|sound|"          >> "$QP"
done
for c in none anon_high anon_low; do echo "s2|$c|grad_noise_g4|--swap_options" >> "$Q"; done
log "queued $(wc -l < "$Q") single + $(wc -l < "$QP") paired"

outfile(){ local sfx=""; [ -n "$4" ] && sfx="_swap"
  case "$1" in
    s2) echo "outputs/s2_$3_$2${sfx}_v1.jsonl";; q3) echo "outputs/s2q3_$3_$2${sfx}_v1.jsonl";;
    qwen2audio) echo "outputs/nm_qwen2audio_$3_$2${sfx}_v1.jsonl";;
    af3) echo "outputs/nm_af3_$3_$2${sfx}_v1.jsonl";; vl) echo "outputs/s2vl_$3_$2${sfx}_v1.jsonl";;
  esac; }
launch(){ local kind=$1 cond=$2 attr=$3 flag=$4 g=$5 lg=$6
  case "$kind" in
    s2) CUDA_VISIBLE_DEVICES=$g python3 scripts/eval_stage2_factorial_v1.py --attr "$attr" --cond "$cond" $flag > "$lg" 2>&1;;
    q3) CUDA_VISIBLE_DEVICES=$g python3 scripts/eval_stage2_q3_v1.py --attr "$attr" --cond "$cond" $flag > "$lg" 2>&1;;
    qwen2audio) CUDA_VISIBLE_DEVICES=$g python3 scripts/eval_newmodels_v1.py --model qwen2audio --attr "$attr" --cond "$cond" $flag > "$lg" 2>&1;;
    af3) CUDA_VISIBLE_DEVICES=$g bash scripts/run_af3_v1.sh --attr "$attr" --cond "$cond" $flag > "$lg" 2>&1;;
    vl) CUDA_VISIBLE_DEVICES=$g python3 scripts/eval_stage2_vllama_v1.py --attr "$attr" --cond "$cond" $flag > "$lg" 2>&1;;
  esac; }
pop(){ exec 9>"$LK"; flock 9; local l; l=$(head -1 "$1"); [ -n "$l" ] && sed -i '1d' "$1"; flock -u 9; exec 9>&-; echo "$l"; }
worker(){ local wid=$1 g=$2
  while :; do
    local job; job=$(pop "$Q"); [ -z "$job" ] && break
    IFS='|' read -r kind cond attr flag <<< "$job"
    local out; out=$(outfile "$kind" "$cond" "$attr" "$flag")
    local lg="${out%_v1.jsonl}.log" n=0
    for t in 1 2 3; do
      [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ] && { n=400; break; }
      while :; do f=$(nvidia-smi -i "$g" --query-gpu=memory.free --format=csv,noheader,nounits); [ "$f" -gt 28000 ] && break; sleep 45; done
      launch "$kind" "$cond" "$attr" "$flag" "$g" "$lg"
      n=$(wc -l < "$out" 2>/dev/null || echo 0); [ "$n" -ge 400 ] && break; sleep 20
    done
    log "w$wid $kind/$attr/$cond$flag -> $n"
  done; }
worker 1 0 & worker 2 0 & worker 3 1 & worker 4 1 & worker 5 2 & worker 6 3 &
wait
log "single-card done; Qwen3 pair starting"
while :; do
  job=$(pop "$QP"); [ -z "$job" ] && break
  IFS='|' read -r kind cond attr flag <<< "$job"
  out=$(outfile "$kind" "$cond" "$attr" "$flag"); lg="${out%_v1.jsonl}.log"; n=0
  for t in 1 2 3; do
    [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ] && { n=400; break; }
    while :; do a=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits)
      b=$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits)
      [ "$a" -gt 40000 ] && [ "$b" -gt 40000 ] && break; sleep 60; done
    launch "$kind" "$cond" "$attr" "$flag" "0,1" "$lg"
    n=$(wc -l < "$out" 2>/dev/null || echo 0); [ "$n" -ge 400 ] && break; sleep 20
  done
  log "pair $kind/$attr/$cond -> $n"
done
log "P0 FIX DONE"
