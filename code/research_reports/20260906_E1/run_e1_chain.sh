#!/bin/bash
# E1 chain: Qwen2-Audio (nm runner) then Qwen2.5-Omni (s2 runner); 6 conds x 2 orders each. GPUs 0-3 only.
set -u
cd /workspace/svhalluc/claude_track2
S=research_reports/20260906_E1/e1_status.txt
log(){ echo "[$(date -u +%H:%M)Z] $*" | tee -a "$S"; }
pick_gpu(){ nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
  | awk -F', ' -v n="$1" '$1<4 {free=$3-$2; if (free>n) print free, $1}' | sort -rn | head -1 | awk '{print $2}'; }
run_one(){ # label need out cmd...
  local label=$1 need=$2 out=$3; shift 3
  local n=0; [ -f "$out" ] && n=$(wc -l < "$out"); [ "$n" -ge 400 ] && { log "skip $label ($n)"; return 0; }
  for attempt in 1 2 3; do
    g=""; while [ -z "$g" ]; do g=$(pick_gpu "$need"); [ -z "$g" ] && sleep 60; done
    log "$label attempt $attempt GPU $g"
    LOGF=research_reports/20260906_E1/logs/${label}.log
    CUDA_VISIBLE_DEVICES=$g PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$@" > "$LOGF" 2>&1 &
    JP=$!
    while kill -0 $JP 2>/dev/null; do sleep 60; sk=$(grep -c '^\[skip\]\|^SKIP' "$LOGF" 2>/dev/null || echo 0)
      if [ "$sk" -ge 10 ]; then log "$label poisoned ($sk skips) -> kill"; kill $JP; sleep 5; kill -9 $JP 2>/dev/null; break; fi; done
    wait $JP 2>/dev/null
    n=0; [ -f "$out" ] && n=$(wc -l < "$out")
    if [ "$n" -ge 400 ]; then log "$label OK ($n)"; return 0; fi
    log "$label incomplete ($n), retry"; sleep 120
  done
  log "$label FAILED"; return 1
}
mkdir -p research_reports/20260906_E1/logs
for cond in claimconf_high claimconf_low detconf_high detconf_low readerconf_high readerconf_low; do
  for sw in "" "--swap_options"; do t=""; [ -n "$sw" ] && t="_swap"
    run_one "q2a_${cond}${t}" 26000 outputs/nm_qwen2audio_sound_${cond}${t}_v1.jsonl \
      python3 research_reports/20260906_E1/eval_e1_nm_v1.py --model qwen2audio --attr sound --cond $cond $sw
  done
done
log "Q2A E1 DONE"
for cond in claimconf_high claimconf_low detconf_high detconf_low readerconf_high readerconf_low; do
  for sw in "" "--swap_options"; do t=""; [ -n "$sw" ] && t="_swap"
    run_one "s2_${cond}${t}" 30000 outputs/s2_sound_${cond}${t}_v1.jsonl \
      python3 research_reports/20260906_E1/eval_e1_s2_v1.py --attr sound --cond $cond $sw
  done
done
log "E1 CHAIN DONE"
