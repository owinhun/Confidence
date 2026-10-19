#!/usr/bin/env bash
# Everything the v10 review asks for that needs GPU, chained behind wave9b.
#
#   cascadex_plain / cascade2x_plain   The kill test. v10 argues that packaging, not forwarding,
#       is what costs -- but it argues it from our own true/false branches. Only the detector's
#       actual mixture, sent as a plain sentence, says whether bare forwarding pays at the
#       accuracy this tagger has. Its lowest controlled break-even (0.753) sits just above the
#       detector (0.748), so the honest expectation is that it does not, and the paper should
#       say which.
#
#   ladder swap on cs_*                Two of the five compression cells could not be
#       order-averaged because the ladder script had no --swap_options. It does now.
#
#   colour block-format                The packaging result is measured on sound-event only.
#       If it holds on a visual attribute it is a property of the interface rather than of audio.
set -uo pipefail
cd /workspace/svhalluc/claude_track2
Q=/root/.claude/jobs/65f6dcfb/tmp/w10.txt
LOCK=/root/.claude/jobs/65f6dcfb/tmp/w10.lock
LOG=outputs/wave10_log.txt

while ! grep -q "wave9b. all done" outputs/wave9_log.txt 2>/dev/null; do sleep 120; done
echo "[wave10] wave9b cleared, starting $(date)" | tee -a "$LOG"
: > "$Q"

# 1. kill test: the detector's own decision as a plain sentence, both detectors, both orders
for spec in "s2 fact sound" "s2q3 q3 sound" "nm_qwen2audio new sound" \
            "nm_af3 af3 sound" "nm_nemotron nemo sound"; do
  set -- $spec
  for c in cascadex_plain cascade2x_plain; do
    for o in none swap; do echo "$1 $2 $3 $c $o" >> "$Q"; done
  done
done
# 2. the two ladder cells that could not be order-averaged
for spec in "qwen25 cs" "qwen3 cs"; do
  set -- $spec
  for c in v_vlow v_high n_015 n_085; do
    for cl in false true; do echo "$1 $2 sound $c $cl" >> "$Q"; done
  done
done
# 3. does the packaging effect hold on a visual attribute?
for spec in "s2 fact" "s2q3 q3" "s2vl vl" "nm_nemotron nemo"; do
  set -- $spec
  for c in bare_true detclaim_true detclaim_false; do
    echo "$1 $2 colour $c none" >> "$Q"
  done
done
echo "[queue] $(wc -l < "$Q") jobs" | tee -a "$LOG"

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
      vl)   script=scripts/eval_stage2_vllama_v1.py; pfx=$a ;;
      new)  script=scripts/eval_newmodels_v1.py; pfx=$a; extra="--model ${a#nm_}" ;;
      af3)  script=scripts/run_af3_v1.sh; pfx=$a ;;
      nemo) script=scripts/run_nemotron_v1.sh; pfx=$a; need=65000 ;;
      cs)   script=scripts/eval_confscale_v1.py; pfx="cs_$a"; extra="--model $a --swap_options" ;;
    esac
    if [ "$kind" = cs ]; then
      # the ladder takes --claim rather than a separate condition name
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
        >> "outputs/w10_${pfx}_${attr}_${cond}${tag}.out" 2>&1 || true
    else
      CUDA_VISIBLE_DEVICES=$gpu python3 "$script" --attr "$attr" --cond "$cond" $extra \
        >> "outputs/w10_${pfx}_${attr}_${cond}${tag}.out" 2>&1 || true
    fi
    echo "[gpu$gpu] $(date +%H:%M) done  $pfx $attr $cond $five rows=$( [ -f "$out" ] && wc -l < "$out" || echo 0)" | tee -a "$LOG"
  done
  echo "[gpu$gpu] empty" | tee -a "$LOG"
}

for g in 0 1 2 3; do worker "$g" & done
wait
echo "[wave10] all done $(date)" | tee -a "$LOG"
