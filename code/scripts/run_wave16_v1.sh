#!/usr/bin/env bash
# The control the design is still missing: epistemic wording that is not about this claim.
#
# Confidence, Reliability and Certainty all say how much to believe the claim in the block.
# Priority, Urgency and Metric-X say nothing epistemic at all. So the 2.1-3.9x gap is open to
# one more reading -- that the model responds to a field carrying *any* meaning, and the
# non-epistemic controls are simply meaningless. Metric-X blunts that but is itself opaque.
#
# `Prior: high` fills the missing cell of the 2x2: same block, same graded value, epistemic
# vocabulary, but scoped to the class base rate rather than to this clip's claim. If it behaves
# like Confidence, the effect is about epistemic register. If it behaves like Priority, the
# effect needs the epistemic content to be aimed at the claim -- which is the paper's thesis.
#
# All ten primary cells, both answer orders, so the result can be quoted on the same set as
# the headline contrast.
set -uo pipefail
cd /workspace/svhalluc/claude_track2
Q=/root/.claude/jobs/65f6dcfb/tmp/w16.txt
LOCK=/root/.claude/jobs/65f6dcfb/tmp/w16.lock
LOG=outputs/wave16_log.txt
: > "$Q"
for spec in "s2 fact sound" "s2 fact colour" "s2 fact grad_noise_g4" \
            "s2q3 q3 sound" "s2q3 q3 noise" "s2q3 q3 colour" \
            "s2vl vl colour" "nm_qwen2audio new sound" \
            "nm_nemotron nemo sound" "nm_nemotron nemo colour"; do
  set -- $spec
  for c in prior_high prior_low; do
    for o in none swap; do echo "$1 $2 $3 $c $o" >> "$Q"; done
  done
done
echo "[queue] $(wc -l < "$Q") jobs $(date)" | tee -a "$LOG"

take() { flock "$LOCK" bash -c "l=\$(head -1 $Q); [ -n \"\$l\" ] && sed -i 1d $Q; echo \"\$l\""; }

worker() {
  local gpu=$1
  while :; do
    local job; job=$(take); [ -z "$job" ] && break
    set -- $job
    local pfx=$1 kind=$2 attr=$3 cond=$4 order=$5
    local script need=30000 extra="" tag=""
    case $kind in
      fact) script=scripts/eval_stage2_factorial_v1.py ;;
      q3)   script=scripts/eval_stage2_q3_v1.py; need=45000 ;;
      vl)   script=scripts/eval_stage2_vllama_v1.py ;;
      new)  script=scripts/eval_newmodels_v1.py; extra="--model ${pfx#nm_}" ;;
      nemo) script=scripts/run_nemotron_v1.sh; need=65000 ;;
    esac
    [ "$order" = swap ] && { extra="$extra --swap_options"; tag="_swap"; }
    local out="outputs/${pfx}_${attr}_${cond}${tag}_v1.jsonl"
    if [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ]; then
      echo "[skip] $(basename $out)" | tee -a "$LOG"; continue
    fi
    while :; do
      local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu")
      [ "$free" -ge "$need" ] && break
      sleep 45
    done
    echo "[gpu$gpu] $(date +%H:%M) start $pfx $attr $cond $order" | tee -a "$LOG"
    if [[ $script == *.sh ]]; then
      CUDA_VISIBLE_DEVICES=$gpu bash "$script" --attr "$attr" --cond "$cond" $extra \
        >> "outputs/w16_${pfx}_${attr}_${cond}${tag}.out" 2>&1 || true
    else
      CUDA_VISIBLE_DEVICES=$gpu python3 "$script" --attr "$attr" --cond "$cond" $extra \
        >> "outputs/w16_${pfx}_${attr}_${cond}${tag}.out" 2>&1 || true
    fi
    echo "[gpu$gpu] $(date +%H:%M) done  $pfx $attr $cond $order rows=$( [ -f "$out" ] && wc -l < "$out" || echo 0)" | tee -a "$LOG"
  done
  echo "[gpu$gpu] empty" | tee -a "$LOG"
}
for g in 0 1 2 3; do worker "$g" & done
wait
echo "[wave16] all done $(date)" | tee -a "$LOG"
