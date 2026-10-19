#!/usr/bin/env bash
# The Nemotron half of wave8, held until wave7's Nemotron worker clears GPU0.
# Split from the 7B half so three idle cards did not sit behind one serial 30B job.
set -uo pipefail
cd /workspace/svhalluc/claude_track2
LOG=outputs/wave8_log.txt
while ! grep -q "wave7. all done" outputs/wave7_log.txt 2>/dev/null; do sleep 120; done
echo "[wave8-nemo] wave7 cleared, starting $(date)" | tee -a "$LOG"
for spec in "n_015 false" "n_015 true" "n_060 false" "n_060 true" "n_085 false" "n_085 true" \
            "v_high false" "v_high true" "bare_true none" "null_priority none"; do
  set -- $spec; cond=$1; cl=$2
  tag=""; [ "$cl" = true ] && tag="_T"
  out="outputs/nm_nemotron_sound_${cond}${tag}_v1.jsonl"
  if [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ]; then
    echo "[skip] $(basename $out)" | tee -a "$LOG"; continue
  fi
  cf="--claim $cl"; case $cond in bare_true|null_priority) cf="" ;; esac
  gpu=""
  while [ -z "$gpu" ]; do
    for g in 0 1 2 3; do
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g")
      if [ "$free" -ge 65000 ]; then gpu=$g; break; fi
    done
    [ -z "$gpu" ] && sleep 90
  done
  echo "[gpu$gpu] $(date +%H:%M) start nemotron $cond claim=$cl" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$gpu bash scripts/run_nemotron_v1.sh --attr sound --cond "$cond" $cf \
    >> "outputs/w8_nm_nemotron_${cond}${tag}.out" 2>&1 || true
  echo "[gpu$gpu] $(date +%H:%M) done  nemotron $cond claim=$cl rows=$( [ -f "$out" ] && wc -l < "$out" || echo 0)" | tee -a "$LOG"
done
echo "[wave8-nemo] all done $(date)" | tee -a "$LOG"
