#!/usr/bin/env bash
# Fill AF3's Prior cell. wave16 ran Prior on the ten primary cells only, so the sensitivity row
# was the one blank in an otherwise complete column -- a gap a reader notices before they notice
# what the column shows. AF3 is reported as sensitivity either way; having the value simply
# removes the question.
set -uo pipefail
cd /workspace/svhalluc/claude_track2
Q=/root/.claude/jobs/65f6dcfb/tmp/w17.txt
LOCK=/root/.claude/jobs/65f6dcfb/tmp/w17.lock
LOG=outputs/wave17_log.txt
: > "$Q"
for c in prior_high prior_low; do
  for o in none swap; do echo "$c $o" >> "$Q"; done
done
echo "[queue] $(wc -l < "$Q") jobs $(date)" | tee -a "$LOG"
take() { flock "$LOCK" bash -c "l=\$(head -1 $Q); [ -n \"\$l\" ] && sed -i 1d $Q; echo \"\$l\""; }
worker() {
  local gpu=$1
  while :; do
    local job; job=$(take); [ -z "$job" ] && break
    set -- $job; local cond=$1 order=$2 extra="" tag=""
    [ "$order" = swap ] && { extra="--swap_options"; tag="_swap"; }
    local out="outputs/nm_af3_sound_${cond}${tag}_v1.jsonl"
    if [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ]; then
      echo "[skip] $(basename $out)" | tee -a "$LOG"; continue
    fi
    while :; do
      local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu")
      [ "$free" -ge 30000 ] && break
      sleep 45
    done
    echo "[gpu$gpu] $(date +%H:%M) start af3 sound $cond $order" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=$gpu bash scripts/run_af3_v1.sh --attr sound --cond "$cond" $extra \
      >> "outputs/w17_af3_${cond}${tag}.out" 2>&1 || true
    echo "[gpu$gpu] $(date +%H:%M) done  af3 sound $cond $order rows=$( [ -f "$out" ] && wc -l < "$out" || echo 0)" | tee -a "$LOG"
  done
  echo "[gpu$gpu] empty" | tee -a "$LOG"
}
for g in 0 1 2 3; do worker "$g" & done
wait
echo "[wave17] all done $(date)" | tee -a "$LOG"
