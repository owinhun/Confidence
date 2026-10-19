#!/bin/bash
# Follow-on to the four gates now running: finish the swapped-option halves, judge every cell
# under the same rule the withdrawn VideoLLaMA sound cell was judged by, and start Tier 1 on
# whatever survives.
#
# The gate is applied to BOTH answer orders on purpose. VideoLLaMA's sound cell looked fine in
# one order and collapsed to chance in the other, and it was withdrawn for that. Holding a new
# backbone to a weaker standard than the one that removed an existing cell would be the kind of
# asymmetry a reviewer is right to punish, so the swapped pass is a prerequisite, not an extra.
#
# Tier 1 is the nine conditions that feed the paper's three headline tables:
#   anon_{high,low}          the override contrast
#   priority_{high,low}      the non-epistemic control, and the "+ claim" rung's denominator
#   srconly_{high,low}       the rung that makes "+ claim" one component wide
#   true_high                the true branch, for break-even
#   priority_true_{high,low} the claim-aligned contrast for a field declaring no confidence
# Tiers 2 and 3 (synonyms, source factor, the five-level ladder) are left for a later pass so a
# model that passes contributes to the main tables within hours rather than a day.
set -u
cd /workspace/svhalluc/claude_track2
S=outputs/stage2_status.txt
K(){ python3 -c "import datetime;print((datetime.datetime.utcnow()+datetime.timedelta(hours=9)).strftime('%m-%d %H:%M'))"; }
log(){ echo "[KST $(K)] $*" >> "$S"; }

TIER1=(anon_high anon_low priority_high priority_low srconly_high srconly_low
       priority_true_high priority_true_low)

launch(){ # kind attr cond gpu extra -> rows
  local kind=$1 attr=$2 cond=$3 g=$4 extra=${5:-}
  local pfx out sfx=""
  [ "$kind" = nemotron ] && pfx=nm_nemotron || pfx=gm_$kind
  [[ "$extra" == *swap* ]] && sfx="_swap"
  [[ "$extra" == *"--claim true"* ]] && sfx="_T$sfx"
  out="outputs/${pfx}_${attr}_${cond}${sfx}_v1.jsonl"
  if [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ]; then echo 400; return; fi
  while :; do f=$(nvidia-smi -i "$g" --query-gpu=memory.free --format=csv,noheader,nounits)
    [ "$f" -gt 30000 ] && break; sleep 60; done
  if [ "$kind" = nemotron ]; then
    CUDA_VISIBLE_DEVICES=$g timeout 10800 bash scripts/run_nemotron_v1.sh \
      --attr "$attr" --cond "$cond" $extra > "${out%_v1.jsonl}.log" 2>&1
  else
    CUDA_VISIBLE_DEVICES=$g timeout 10800 python3 scripts/eval_generic_v1.py --model "$kind" \
      --attr "$attr" --cond "$cond" $extra > "${out%_v1.jsonl}.log" 2>&1
  fi
  wc -l < "$out" 2>/dev/null || echo 0
}

verdict(){ # pfx attr
  python3 - "$1" "$2" << 'PY'
import json, os, sys
pfx, attr = sys.argv[1], sys.argv[2]
def load(t):
    p = f"outputs/{pfx}_{attr}_none{t}_v1.jsonl"
    if not os.path.exists(p): return None
    rs = [json.loads(l) for l in open(p) if l.strip()]
    return rs if len(rs) >= 400 else None
def stat(rs):
    a = sum(x["correct"] for x in rs) / len(rs)
    r = [sum(x["correct"] for x in rs if x["label"] == L) /
         max(1, sum(1 for x in rs if x["label"] == L)) for L in ("A", "B")]
    return a, r
o, s = load(""), load("_swap")
if o is None: print("FAIL:no-baseline"); raise SystemExit
ao, ro = stat(o)
if s is None: print(f"PENDING-SWAP A0={ao*100:.1f} rec={ro[0]*100:.0f}/{ro[1]*100:.0f}"); raise SystemExit
as_, rs_ = stat(s)
avg = (ao + as_) / 2
tag = f"A0={ao*100:.1f}/{as_*100:.1f} avg={avg*100:.1f} rec={ro[0]*100:.0f}/{ro[1]*100:.0f} swaprec={rs_[0]*100:.0f}/{rs_[1]*100:.0f}"
if min(ro) < 0.60 or min(rs_) < 0.60: print(f"FAIL:one-sided {tag}")
elif avg < 0.60: print(f"FAIL:floor {tag}")
elif avg > 0.95: print(f"CEILING {tag}")
else: print(f"PASS {tag}")
PY
}

# wait for the four gates already in flight
while pgrep -f 'eval_[n]emotron_v1.py|eval_[g]eneric_v1.py' > /dev/null; do sleep 120; done
log "initial gates finished; completing the swapped halves"

# the two swapped passes that are still missing, in parallel on the freed cards
( n=$(launch nemotron colour none 0 "--swap_options"); log "nemotron/colour gate(swap) -> $n" ) &
( n=$(launch meralion sound none 1 "--swap_options"); log "meralion/sound gate(swap) -> $n" ) &
wait
log "swapped gates done"

GPU=0
for cell in "nemotron sound" "nemotron colour" "meralion sound"; do
  set -- $cell; KIND=$1; ATTR=$2
  pfx=nm_nemotron; [ "$KIND" != nemotron ] && pfx=gm_$KIND
  V=$(verdict "$pfx" "$ATTR"); log "VERDICT $KIND/$ATTR :: $V"
  case "$V" in
    PASS*|CEILING*)
      log "$KIND/$ATTR -> Tier 1 on GPU $GPU"
      ( for C in "${TIER1[@]}"; do
          n=$(launch "$KIND" "$ATTR" "$C" "$GPU"); log "  $KIND/$ATTR/$C -> $n"
        done
        n=$(launch "$KIND" "$ATTR" true_high "$GPU" "--claim true"); log "  $KIND/$ATTR/true_high -> $n"
        log "$KIND/$ATTR Tier 1 complete" ) &
      GPU=$((GPU + 1));;
    *) log "$KIND/$ATTR gated out; recorded in the eligibility flow";;
  esac
done
wait
log "STAGE 2 COMPLETE"
