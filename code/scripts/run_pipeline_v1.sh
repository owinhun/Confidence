#!/bin/bash
# Continuous generalisation pipeline: one model at a time, gate first, battery only if the gate
# is passed, and never stop because one model failed.
#
# The ordering matters. A full battery is ~25 runs; the gate is 1 run per attribute. Running the
# gate first means a model that cannot judge the property at all costs minutes instead of hours,
# and a model that fails is still reported -- the eligibility flow in the paper counts attempted
# cells, so a documented failure strengthens the table rather than being wasted work.
#
# Each model is wrapped so that a crashed load, a missing checkpoint or an OOM logs and moves on.
# Weights that are still downloading are waited for, up to a bounded time, then skipped.
set -u
cd /workspace/svhalluc/claude_track2
S=outputs/pipeline_status.txt
K(){ python3 -c "import datetime;print((datetime.datetime.utcnow()+datetime.timedelta(hours=9)).strftime('%m-%d %H:%M'))"; }
log(){ echo "[KST $(K)] $*" >> "$S"; }

# model | launcher | attributes to gate | gpu spec
# launcher is either a generic-harness model name or a dedicated script tag
MODELS=(
  "vllama2_mistral|generic|colour|0"
  "video_llava|generic|colour|1"
  "ultravox|generic|sound|2"
  "llava_next_video|generic|colour|3"
  "nemotron|nemotron|sound colour|0,1"
)

# the full battery, in the order the paper needs it: the two contrasts that carry the headline
# first, so a model that dies midway still yields the primary numbers
BATTERY=(none anon_high anon_low true_high priority_high priority_low
         priority_true_high priority_true_low srconly_high srconly_low null_block
         expert_high expert_low reliab_high reliab_low certain_high certain_low)

run_one(){ # model launcher attr cond gpu extra
  local m=$1 l=$2 attr=$3 cond=$4 g=$5 extra=${6:-}
  local pfx out
  case "$l" in
    generic)  pfx="gm_$m";;
    nemotron) pfx="nm_nemotron";;
  esac
  local sfx=""; [[ "$extra" == *swap* ]] && sfx="_swap"
  out="outputs/${pfx}_${attr}_${cond}${sfx}_v1.jsonl"
  [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ] && { echo 400; return; }
  while :; do
    local free; free=$(nvidia-smi -i "${g%%,*}" --query-gpu=memory.free --format=csv,noheader,nounits)
    [ "$free" -gt 28000 ] && break; sleep 60
  done
  if [ "$l" = generic ]; then
    CUDA_VISIBLE_DEVICES=$g timeout 7200 python3 scripts/eval_generic_v1.py --model "$m" \
      --attr "$attr" --cond "$cond" $extra > "${out%_v1.jsonl}.log" 2>&1
  else
    CUDA_VISIBLE_DEVICES=$g timeout 14400 bash scripts/run_nemotron_v1.sh \
      --attr "$attr" --cond "$cond" $extra > "${out%_v1.jsonl}.log" 2>&1
  fi
  wc -l < "$out" 2>/dev/null || echo 0
}

gate_verdict(){ # prefix attr -> PASS / FAIL:<reason>
  python3 - "$1" "$2" << 'PY'
import json, os, sys
pfx, attr = sys.argv[1], sys.argv[2]
def load(t):
    p = f"outputs/{pfx}_{attr}_none{t}_v1.jsonl"
    if not os.path.exists(p): return None
    rs = [json.loads(l) for l in open(p) if l.strip()]
    return rs if len(rs) >= 400 else None
r = load("")
if r is None: print("FAIL:no-baseline"); raise SystemExit
acc = sum(x["correct"] for x in r) / len(r)
rec = [sum(x["correct"] for x in r if x["label"] == L) /
       max(1, sum(1 for x in r if x["label"] == L)) for L in ("A", "B")]
s = load("_swap")
extra = ""
if s is not None:
    sa = sum(x["correct"] for x in s) / len(s)
    sr = [sum(x["correct"] for x in s if x["label"] == L) /
          max(1, sum(1 for x in s if x["label"] == L)) for L in ("A", "B")]
    extra = f" swapA0={sa*100:.1f} swaprec={sr[0]*100:.0f}/{sr[1]*100:.0f}"
    if min(sr) < 0.60: print(f"FAIL:swap-recall A0={acc*100:.1f}{extra}"); raise SystemExit
if acc < 0.60: print(f"FAIL:floor A0={acc*100:.1f}{extra}")
elif min(rec) < 0.60: print(f"FAIL:one-sided A0={acc*100:.1f} rec={rec[0]*100:.0f}/{rec[1]*100:.0f}{extra}")
elif acc > 0.95: print(f"CEILING A0={acc*100:.1f}{extra}")
else: print(f"PASS A0={acc*100:.1f} rec={rec[0]*100:.0f}/{rec[1]*100:.0f}{extra}")
PY
}

# do not fight the P0 fix for cards
while [ -s /root/.claude/jobs/65f6dcfb/tmp/p0fix_queue ] || \
      [ -s /root/.claude/jobs/65f6dcfb/tmp/p0fix_pair_queue ]; do sleep 120; done
log "P0 queue clear; generalisation pipeline starting"

for entry in "${MODELS[@]}"; do
  IFS='|' read -r M L ATTRS G <<< "$entry"
  log "=== $M ($L) on GPU $G ==="
  pfx="gm_$M"; [ "$L" = nemotron ] && pfx="nm_nemotron"

  # smoke test: three clips, so a broken adapter surfaces in a minute rather than an hour
  if [ "$L" = generic ]; then
    CUDA_VISIBLE_DEVICES=$G timeout 2400 python3 scripts/eval_generic_v1.py --model "$M" \
      --attr "$(echo $ATTRS | awk '{print $1}')" --cond none --limit 3 \
      > "outputs/${pfx}_smoke.log" 2>&1
  else
    CUDA_VISIBLE_DEVICES=$G timeout 3600 bash scripts/run_nemotron_v1.sh \
      --attr "$(echo $ATTRS | awk '{print $1}')" --cond none --limit 3 \
      > "outputs/${pfx}_smoke.log" 2>&1
  fi
  # "[done] n=0" means every clip raised; require at least one scored item
  if ! grep -qE "\[done\] n=[1-9]" "outputs/${pfx}_smoke.log" 2>/dev/null; then
    log "$M SMOKE FAILED -- $(tail -2 outputs/${pfx}_smoke.log 2>/dev/null | tr '\n' ' ' | cut -c1-160)"
    log "$M skipped; continuing"
    continue
  fi
  log "$M smoke ok"

  for A in $ATTRS; do
    n=$(run_one "$M" "$L" "$A" none "$G"); log "$M/$A gate -> $n rows"
    [ "$n" -lt 400 ] && { log "$M/$A gate incomplete; skipping attribute"; continue; }
    n=$(run_one "$M" "$L" "$A" none "$G" "--swap_options"); log "$M/$A gate(swap) -> $n rows"
    V=$(gate_verdict "$pfx" "$A"); log "$M/$A VERDICT $V"
    case "$V" in
      PASS*|CEILING*)
        for C in "${BATTERY[@]}"; do
          [ "$C" = none ] && continue
          n=$(run_one "$M" "$L" "$A" "$C" "$G"); log "  $M/$A/$C -> $n"
        done
        n=$(run_one "$M" "$L" "$A" true_high "$G"); log "  $M/$A/true_high -> $n"
        log "$M/$A battery complete";;
      *) log "$M/$A gated out; no battery run (recorded in the eligibility flow)";;
    esac
  done
  log "=== $M finished ==="
done
log "PIPELINE COMPLETE"
