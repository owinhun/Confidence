#!/usr/bin/env bash
# Manipulation check for the Audio Flamingo 3 null (open item H).
#
# AF3's override rate is 49.3 at high and 49.2 at low -- flat. A null like that reads two ways:
# the model never attends the metadata block, or it attends and does not integrate what it
# reads. Asking it to report what the message says, rather than what the clip contains, tells
# them apart: the readback needs only that the block was read.
#
# Qwen2-Audio is the positive control. It is the same size and the same modality and it shows a
# large effect (63.0 vs 39.8), so its readback score is what "attends the block" looks like on
# this data. Without it a low AF3 score would have no reference.
#
# Runs alongside wave4 rather than after it: the gate is free memory, so a job only starts on a
# GPU that can hold it, and a 7B model beside another 7B model fits in 80 GB.
set -uo pipefail
cd /workspace/svhalluc/claude_track2
LOG=outputs/readback_log.txt

run_one() {
  local pfx=$1 cond=$2 gpu=$3
  local out="outputs/${pfx}_sound_${cond}_readback_v1.jsonl"
  if [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ]; then
    echo "[skip] $out" | tee -a "$LOG"; return
  fi
  while :; do
    local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu")
    [ "$free" -ge 30000 ] && break
    sleep 120
  done
  local script=scripts/eval_newmodels_v1.py extra="--model ${pfx#nm_}"
  [ "$pfx" = nm_af3 ] && { script=scripts/eval_af3_v1.py; extra=""; }
  echo "[gpu$gpu] $(date +%H:%M) start $pfx $cond readback" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$gpu python3 "$script" --attr sound --cond "$cond" --readback $extra \
    >> "outputs/readback_${pfx}_${cond}.out" 2>&1 || true
  echo "[gpu$gpu] $(date +%H:%M) done  $pfx $cond rows=$( [ -f "$out" ] && wc -l < "$out" || echo 0)" | tee -a "$LOG"
}

for m in nm_af3 nm_qwen2audio; do
  for c in anon_high true_high; do
    run_one "$m" "$c" 0
  done
done
echo "[readback] all done $(date)" | tee -a "$LOG"
