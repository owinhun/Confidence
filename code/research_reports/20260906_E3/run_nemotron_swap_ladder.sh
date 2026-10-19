#!/bin/bash
# E3: Nemotron/sound missing swapped-order rungs. Needs >=65GB exclusive (log 6251: no co-tenant).
set -u
cd /workspace/svhalluc/claude_track2
S=research_reports/20260906_E3/nemotron_status.txt
log(){ echo "[$(date -u +%H:%M)Z] $*" | tee -a "$S"; }
pick_gpu(){ nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits \
  | awk -F', ' '$1<4 {free=$3-$2; if (free>71000 && $4<5) print free, $1}' | sort -rn | head -1 | awk '{print $2}'; }
JOBS="v_vlow:true v_mod:false v_mod:true v_vhigh:false v_vhigh:true"
for j in $JOBS; do cond=${j%%:*}; claim=${j##*:}
  tag=""; [ "$claim" = true ] && tag="_T"
  out=outputs/nm_nemotron_sound_${cond}${tag}_swap_v1.jsonl
  n=0; [ -f "$out" ] && n=$(wc -l < "$out"); [ "$n" -ge 400 ] && { log "skip $cond $claim ($n)"; continue; }
  [ "$n" -gt 0 ] && [ "$n" -lt 400 ] && log "resuming $cond $claim from $n rows"
  for attempt in 1 2 3; do
    g=""; while [ -z "$g" ]; do g=$(pick_gpu); [ -z "$g" ] && sleep 45; done
    log "nemotron $cond $claim attempt $attempt GPU $g"
    LOGF=outputs/nm_nemotron_sound_${cond}${tag}_swap.log
    CUDA_VISIBLE_DEVICES=$g PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash scripts/run_nemotron_v1.sh \
      --attr sound --cond $cond --claim $claim --swap_options > "$LOGF" 2>&1 &
    JP=$!
    # watchdog: a co-tenant arriving during the load leaves a half-resident model that fails every
    # clip with a device-side assert (log 6242 incident); kill such a run early instead of skipping 400 clips
    while kill -0 $JP 2>/dev/null; do
      sleep 60
      sk=$(grep -c '^\[skip\]' "$LOGF" 2>/dev/null || echo 0)
      if [ "$sk" -ge 10 ]; then log "nemotron $cond $claim poisoned ($sk skips) -> kill"; kill $JP; sleep 5; kill -9 $JP 2>/dev/null; break; fi
    done
    wait $JP 2>/dev/null
    n=0; [ -f "$out" ] && n=$(wc -l < "$out")
    if [ "$n" -ge 400 ]; then log "nemotron $cond $claim OK ($n)"; break; fi
    log "nemotron $cond $claim incomplete ($n), retry"; sleep 300
  done
done
log "NEMOTRON CHAIN DONE"
