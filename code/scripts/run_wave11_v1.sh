#!/usr/bin/env bash
# Re-runs forced by the wave10 audit. Three independent failures, one of them a correctness bug.
#
#   detclaim_true / detclaim_false   The claim was built from PROP[attr](cond.endswith("true"))
#       rather than from the clip's label, so both branches were 50% true -- constant "always
#       present" / "always absent" detectors. Every break-even in Table 4's pi* column came from
#       that pair. Fixed and re-run here; the old outputs are kept under outputs/buggy_detclaim/.
#
#   cs_* swapped ladder              eval_confscale_v1.py consulted args.swap_options without
#       declaring it, so all 16 swapped jobs died at argument parsing and the runner's `|| true`
#       recorded them as done with zero rows.
#
#   nemotron cascade2x_plain         246 clips skipped on per-clip OOM while two runners shared
#       a GPU. The script resumes, so this only fills the gap.
#
# One runner this time: wave9c and wave10 both pinned workers to GPUs 0/2/3, which is what put
# two jobs on one card and caused the OOMs in the first place.
set -uo pipefail
cd /workspace/svhalluc/claude_track2
Q=/root/.claude/jobs/65f6dcfb/tmp/w11.txt
LOCK=/root/.claude/jobs/65f6dcfb/tmp/w11.lock
LOG=outputs/wave11_log.txt
: > "$Q"

# 1. the corrected claim branches, both orders, all five cells
for spec in "s2 fact" "s2q3 q3" "nm_qwen2audio new" "nm_af3 af3" "nm_nemotron nemo"; do
  set -- $spec
  for c in detclaim_true detclaim_false; do
    for o in none swap; do echo "$1 $2 sound $c $o" >> "$Q"; done
  done
done
# 2. the same on colour, to see whether the packaging result is about audio or about the interface
for spec in "s2 fact" "s2q3 q3" "nm_nemotron nemo"; do
  set -- $spec
  for c in detclaim_true detclaim_false; do echo "$1 $2 colour $c none" >> "$Q"; done
done
# 3. the swapped ladder that never parsed
for m in qwen25 qwen3; do
  for c in v_vlow v_high n_015 n_085; do
    for cl in false true; do echo "$m cs sound $c $cl" >> "$Q"; done
  done
done
# 4. the truncated kill-test cell
echo "nm_nemotron nemo sound cascade2x_plain none" >> "$Q"
echo "[queue] $(wc -l < "$Q") jobs $(date)" | tee -a "$LOG"

take() { flock "$LOCK" bash -c "l=\$(head -1 $Q); [ -n \"\$l\" ] && sed -i 1d $Q; echo \"\$l\""; }

worker() {
  local gpu=$1
  while :; do
    local job; job=$(take); [ -z "$job" ] && break
    set -- $job
    local a=$1 kind=$2 attr=$3 cond=$4 five=$5
    local script need=30000 extra="" pfx out tag=""
    case $kind in
      fact) script=scripts/eval_stage2_factorial_v1.py; pfx=$a ;;
      q3)   script=scripts/eval_stage2_q3_v1.py; pfx=$a; need=45000 ;;
      new)  script=scripts/eval_newmodels_v1.py; pfx=$a; extra="--model ${a#nm_}" ;;
      af3)  script=scripts/run_af3_v1.sh; pfx=$a ;;
      nemo) script=scripts/run_nemotron_v1.sh; pfx=$a; need=65000 ;;
      cs)   script=scripts/eval_confscale_v1.py; pfx="cs_$a"; extra="--model $a --swap_options" ;;
    esac
    if [ "$kind" = cs ]; then
      [ "$five" = true ] && { extra="$extra --claim true"; tag="_T"; }
      tag="${tag}_swap"
    else
      [ "$five" = swap ] && { extra="$extra --swap_options"; tag="_swap"; }
    fi
    out="outputs/${pfx}_${attr}_${cond}${tag}_v1.jsonl"
    if [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ]; then
      echo "[skip] $(basename $out)" | tee -a "$LOG"; continue
    fi
    while :; do
      local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu")
      [ "$free" -ge "$need" ] && break
      sleep 45
    done
    echo "[gpu$gpu] $(date +%H:%M) start $pfx $attr $cond $five" | tee -a "$LOG"
    if [[ $script == *.sh ]]; then
      CUDA_VISIBLE_DEVICES=$gpu bash "$script" --attr "$attr" --cond "$cond" $extra \
        >> "outputs/w11_${pfx}_${attr}_${cond}${tag}.out" 2>&1 || true
    else
      CUDA_VISIBLE_DEVICES=$gpu python3 "$script" --attr "$attr" --cond "$cond" $extra \
        >> "outputs/w11_${pfx}_${attr}_${cond}${tag}.out" 2>&1 || true
    fi
    echo "[gpu$gpu] $(date +%H:%M) done  $pfx $attr $cond $five rows=$( [ -f "$out" ] && wc -l < "$out" || echo 0)" | tee -a "$LOG"
  done
  echo "[gpu$gpu] empty" | tee -a "$LOG"
}

for g in 0 1 2 3; do worker "$g" & done
wait
echo "[wave11] all done $(date)" | tee -a "$LOG"
