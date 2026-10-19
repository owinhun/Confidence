#!/bin/bash
# Tier 1 for every cell that passed the gate, spread across whatever cards are free.
#
# The previous runner gave each cell one GPU, so nine conditions ran end to end -- about seven
# hours per cell. The conditions are independent, so a shared work queue with one worker per
# free card cuts that to roughly a third. A flock'd queue is what keeps two workers from
# starting the same condition and appending to one jsonl, which has bitten this project before.
#
# Cells are added as their gates clear: sound is already PASS (A0 92.0/91.0, recall 93/91 and
# 89/93), colour is still finishing its swapped half and is appended only if it passes the same
# two-order rule.
set -u
cd /workspace/svhalluc/claude_track2
S=outputs/tier1_status.txt
Q=/root/.claude/jobs/65f6dcfb/tmp/tier1_queue
LK=/root/.claude/jobs/65f6dcfb/tmp/tier1.lock
K(){ python3 -c "import datetime;print((datetime.datetime.utcnow()+datetime.timedelta(hours=9)).strftime('%m-%d %H:%M'))"; }
log(){ echo "[KST $(K)] $*" >> "$S"; }
TIER1=(anon_high anon_low priority_high priority_low srconly_high srconly_low
       priority_true_high priority_true_low)

: > "$Q"
for C in "${TIER1[@]}"; do echo "nemotron|sound|$C|" >> "$Q"; done
echo "nemotron|sound|true_high|--claim true" >> "$Q"
log "queued $(wc -l < "$Q") conditions for nemotron/sound"

pop(){ exec 9>"$LK"; flock 9; local l; l=$(head -1 "$Q"); [ -n "$l" ] && sed -i '1d' "$Q"; flock -u 9; exec 9>&-; echo "$l"; }

worker(){ local wid=$1 g=$2
  while :; do
    local job; job=$(pop); [ -z "$job" ] && break
    IFS='|' read -r kind attr cond extra <<< "$job"
    local pfx sfx=""
    [ "$kind" = nemotron ] && pfx=nm_nemotron || pfx=gm_$kind
    [[ "$extra" == *"claim true"* ]] && sfx="_T"
    local out="outputs/${pfx}_${attr}_${cond}${sfx}_v1.jsonl" n=0
    if [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ]; then log "w$wid $attr/$cond already done"; continue; fi
    while :; do f=$(nvidia-smi -i "$g" --query-gpu=memory.free --format=csv,noheader,nounits)
      [ "$f" -gt 65000 ] && break; sleep 60; done
    if [ "$kind" = nemotron ]; then
      CUDA_VISIBLE_DEVICES=$g timeout 10800 bash scripts/run_nemotron_v1.sh --attr "$attr" --cond "$cond" $extra > "${out%_v1.jsonl}.log" 2>&1
    else
      CUDA_VISIBLE_DEVICES=$g timeout 10800 python3 scripts/eval_generic_v1.py --model "$kind" --attr "$attr" --cond "$cond" $extra > "${out%_v1.jsonl}.log" 2>&1
    fi
    n=$(wc -l < "$out" 2>/dev/null || echo 0)
    log "w$wid $attr/$cond -> $n"
  done; log "worker $wid done"; }

# colour's swapped gate is still running on GPU 0; when it lands, judge it and extend the queue
( while pgrep -f 'eval_[n]emotron_v1.py --attr colour --cond none' > /dev/null; do sleep 60; done
  V=$(python3 - << 'PY'
import json, os, numpy as np
def st(p):
    if not os.path.exists(p): return None
    rs=[json.loads(l) for l in open(p) if l.strip()]
    if len(rs)<400: return None
    return (100*np.mean([r['correct'] for r in rs]),
            [100*np.mean([r['correct'] for r in rs if r['label']==L]) for L in "AB"])
o=st("outputs/nm_nemotron_colour_none_v1.jsonl"); s=st("outputs/nm_nemotron_colour_none_swap_v1.jsonl")
if not o or not s: print("PENDING"); raise SystemExit
avg=(o[0]+s[0])/2
print("PASS" if (min(o[1])>=60 and min(s[1])>=60 and 60<=avg<=95)
      else f"FAIL A0={o[0]:.1f}/{s[0]:.1f} rec={o[1][0]:.0f}/{o[1][1]:.0f} swap={s[1][0]:.0f}/{s[1][1]:.0f}")
PY
)
  log "nemotron/colour VERDICT $V"
  if [[ "$V" == PASS* ]]; then
    exec 9>"$LK"; flock 9
    for C in "${TIER1[@]}"; do echo "nemotron|colour|$C|" >> "$Q"; done
    echo "nemotron|colour|true_high|--claim true" >> "$Q"
    flock -u 9; exec 9>&-
    log "colour passed; 9 more conditions queued"
  fi ) &

worker 1 1 & worker 2 2 & worker 3 3 &
sleep 5
# GPU 0 joins as soon as the colour gate frees it
( while pgrep -f 'eval_[n]emotron_v1.py --attr colour --cond none' > /dev/null; do sleep 60; done
  worker 4 0 ) &
wait
log "TIER 1 COMPLETE"
