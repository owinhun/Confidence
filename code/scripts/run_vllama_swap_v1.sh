#!/bin/bash
# VideoLLaMA2.1's sound cells collapse to chance under swapped options. Its greyscale cells
# carry the two largest contrasts in the paper, so they must be checked the same way before
# either keeping or withdrawing them.
set -u
cd /workspace/svhalluc/claude_track2
S=outputs/vlswap_status.txt
K(){ python3 -c "import datetime;print((datetime.datetime.utcnow()+datetime.timedelta(hours=9)).strftime('%H:%M'))"; }
log(){ echo "[KST $(K)] $*" >> "$S"; }
for c in none anon_high anon_low; do
  out="outputs/s2vl_colour_${c}_swap_v1.jsonl"
  for t in 1 2 3; do
    [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ] && break
    g=""
    while [ -z "$g" ]; do
      for i in 0 1 2 3; do
        f=$(nvidia-smi -i $i --query-gpu=memory.free --format=csv,noheader,nounits)
        [ "$f" -gt 27000 ] && { g=$i; break; }
      done
      [ -z "$g" ] && sleep 60
    done
    CUDA_VISIBLE_DEVICES=$g python3 scripts/eval_stage2_vllama_v1.py --attr colour \
      --cond "$c" --swap_options > "outputs/s2vl_colour_${c}_swap.log" 2>&1
    n=$(wc -l < "$out" 2>/dev/null||echo 0); log "colour/$c try$t [gpu$g] -> $n"
    [ "$n" -ge 400 ] && break
  done
done
log "VLLAMA COLOUR SWAP DONE"
