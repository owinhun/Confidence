#!/usr/bin/env bash
# The Nemotron half of wave4, through the wrapper that injects /data4/pyenv_af3.
#
# Same root cause as the AF3 split: run_wave4_v1.sh invokes the eval scripts directly, and both
# Nemotron and AF3 need transformers 5.x, which the shared environment cannot provide. Nemotron's
# remote code imports `transformers.initialization`; under 4.57 that is an ImportError at load,
# so the job dies before it reads a clip. Two models, one mistake, made twice in one runner.
#
# 65 GB gate: the 30B-A3B checkpoint in BF16 does not share a card with a 7B job.
set -uo pipefail
cd /workspace/svhalluc/claude_track2
LOG=outputs/nemotron_wave4_log.txt

for cond in cascade_mono cascade_pct; do
  out="outputs/nm_nemotron_sound_${cond}_v1.jsonl"
  if [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ]; then
    echo "[skip] $out" | tee -a "$LOG"; continue
  fi
  gpu=""
  while [ -z "$gpu" ]; do
    for g in 0 1 2 3; do
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g")
      if [ "$free" -ge 65000 ]; then gpu=$g; break; fi
    done
    [ -z "$gpu" ] && sleep 120
  done
  echo "[gpu$gpu] $(date +%H:%M) start nemotron $cond" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$gpu bash scripts/run_nemotron_v1.sh --attr sound --cond "$cond" \
    >> "outputs/nemow_${cond}.out" 2>&1 || true
  echo "[gpu$gpu] $(date +%H:%M) done  nemotron $cond rows=$( [ -f "$out" ] && wc -l < "$out" || echo 0)" | tee -a "$LOG"
done
echo "[nemotron] all done $(date)" | tee -a "$LOG"
