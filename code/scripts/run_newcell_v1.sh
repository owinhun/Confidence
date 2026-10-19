#!/bin/bash
# Run the confidence contrast on any graded acoustic cell that the screen found eligible.
#
# Every low-level acoustic cell failed the perceptual-eligibility gate in v5, which is why the
# paper had to say it makes no claim about low-level acoustic properties. The graded screen was
# built to test whether that was a property of the attribute or of the operating point, and the
# answer is the operating point: on Qwen2.5, additive noise goes 51.7 -> 58.8 -> 73.8 -> 86.2
# unaided accuracy from g1 to g4, entering the gate's band at g4.
#
# So the cells that just became askable have to actually be asked. Eligibility is recomputed
# here at run time rather than hard-coded, because the screen is still finishing and the set is
# not known when this is written.
set -u
cd /workspace/svhalluc/claude_track2
S=outputs/newcell_status.txt
Q=/root/.claude/jobs/65f6dcfb/tmp/newcell_queue
LK=/root/.claude/jobs/65f6dcfb/tmp/newcell.lock
K(){ python3 -c "import datetime;print((datetime.datetime.utcnow()+datetime.timedelta(hours=9)).strftime('%H:%M'))"; }
log(){ echo "[KST $(K)] $*" >> "$S"; }

while [ -s /root/.claude/jobs/65f6dcfb/tmp/graded_queue ] || \
      [ -s /root/.claude/jobs/65f6dcfb/tmp/identify_queue ] || \
      [ -s /root/.claude/jobs/65f6dcfb/tmp/identify_pair_queue ]; do sleep 60; done
while pgrep -f "eval_stage2_factorial_v1.py|eval_newmodels_v1.py|eval_af3_v1.py|eval_confscale_v1.py|eval_stage2_vllama_v1.py" > /dev/null; do sleep 60; done
log "screen and identification clear; scoring eligibility"

: > "$Q"
python3 - << 'PY' >> "$Q"
import json, os
MOD = {"s2": "s2", "nm_qwen2audio": "nm_qwen2audio"}
ATTRS = [f"grad_{a}_{g}" for a in ("noise", "lowpass", "reverb") for g in ("g1","g2","g3","g4")]
for base in MOD:
    for attr in ATTRS:
        p = f"outputs/{base}_{attr}_none_v1.jsonl"
        if not os.path.exists(p):
            continue
        rs = [json.loads(l) for l in open(p).read().splitlines() if l.strip()]
        if len(rs) < 400:
            continue
        acc = sum(r["correct"] for r in rs) / len(rs)
        rec = [sum(r["correct"] for r in rs if r["label"] == L) /
               max(1, sum(1 for r in rs if r["label"] == L)) for L in ("A", "B")]
        if 0.60 <= acc <= 0.95 and min(rec) >= 0.60:
            for cond in ("anon_high", "anon_low", "true_high",
                         "priority_high", "priority_low"):
                print(f"{base}|{cond}|{attr}")
PY
log "queued $(wc -l < "$Q") runs on newly eligible acoustic cells"
[ -s "$Q" ] || { log "no cell became eligible; nothing to run"; exit 0; }

pop(){ exec 9>"$LK"; flock 9; local l; l=$(head -1 "$Q"); [ -n "$l" ] && sed -i '1d' "$Q"; flock -u 9; exec 9>&-; echo "$l"; }
worker(){ local wid=$1 g=$2
  while :; do
    local job; job=$(pop); [ -z "$job" ] && break
    IFS='|' read -r base cond attr <<< "$job"
    local out="outputs/${base}_${attr}_${cond}_v1.jsonl" lg n=0
    lg="${out%_v1.jsonl}.log"
    for t in 1 2 3; do
      [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ] && { n=400; break; }
      while :; do f=$(nvidia-smi -i "$g" --query-gpu=memory.free --format=csv,noheader,nounits); [ "$f" -gt 28000 ] && break; sleep 60; done
      if [ "$base" = "nm_qwen2audio" ]; then
        CUDA_VISIBLE_DEVICES=$g python3 scripts/eval_newmodels_v1.py --model qwen2audio --attr "$attr" --cond "$cond" > "$lg" 2>&1
      else
        CUDA_VISIBLE_DEVICES=$g python3 scripts/eval_stage2_factorial_v1.py --attr "$attr" --cond "$cond" > "$lg" 2>&1
      fi
      n=$(wc -l < "$out" 2>/dev/null || echo 0); [ "$n" -ge 400 ] && break; sleep 20
    done
    log "w$wid $base/$attr/$cond -> $n"
  done; log "worker $wid done"; }

worker 1 0 & worker 2 1 & worker 3 2 & worker 4 3 &
wait
log "NEW-CELL CONTRAST DONE"
