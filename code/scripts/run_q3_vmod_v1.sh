#!/bin/bash
set -u
cd /workspace/svhalluc/claude_track2
S=outputs/q3vmod_status.txt
for try in 1 2 3; do
  out=outputs/cs_qwen3_sound_v_mod_T_v1.jsonl
  [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ] && break
  P=""
  while [ -z "$P" ]; do
    for g in 0 1 2 3; do for h in 0 1 2 3; do
      [ "$g" -ge "$h" ] && continue
      f=$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits)
      fh=$(nvidia-smi -i $h --query-gpu=memory.free --format=csv,noheader,nounits)
      [ $((f+fh)) -gt 84000 ] && { P="$g,$h"; break; }
    done; [ -n "$P" ] && break; done
    [ -z "$P" ] && sleep 120
  done
  CUDA_VISIBLE_DEVICES=$P python3 scripts/eval_confscale_v1.py --model qwen3 \
    --attr sound --cond v_mod --claim true > outputs/cs_qwen3_sound_v_mod_T.log 2>&1
  echo "[$(TZ=Asia/Seoul date +%H:%M)] try$try [$P] -> $(wc -l < "$out" 2>/dev/null || echo 0)" >> "$S"
done
echo "[$(TZ=Asia/Seoul date +%H:%M)] V_MOD DONE" >> "$S"
