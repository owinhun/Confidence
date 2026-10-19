#!/usr/bin/env bash
# Second-detector cascade. One scheduler, Nemotron on its own card -- the arrangement that
# stopped it starving behind 7B jobs last time.
set -uo pipefail
cd /workspace/svhalluc/claude_track2
QN=/root/.claude/jobs/65f6dcfb/tmp/w7_nemo.txt
QS=/root/.claude/jobs/65f6dcfb/tmp/w7_small.txt
LOCK=/root/.claude/jobs/65f6dcfb/tmp/w7.lock
LOG=outputs/wave7_log.txt
: > "$QN"; : > "$QS"
for o in none swap; do
  for c in cascadex_bare cascadex_conf cascadex_pct cascade2x_bare cascade2x_conf cascade2x_pct; do
    echo "nemotron $c $o" >> "$QN"
    echo "af3 $c $o" >> "$QS"
    for m in "s2 factorial" "s2q3 q3" "nm_qwen2audio new"; do
      set -- $m; echo "$1 $2 $c $o" >> "$QS"
    done
  done
done
echo "[queue] nemotron $(wc -l < "$QN")  other $(wc -l < "$QS")" | tee -a "$LOG"
take() { flock "$LOCK" bash -c "f=\$1; l=\$(head -1 \"\$f\"); [ -n \"\$l\" ] && sed -i 1d \"\$f\"; echo \"\$l\"" _ "$1"; }
run_job() {
  local gpu=$1; shift
  local a=$1 b=$2 c=$3 d=${4:-}
  local pfx script need extra cond order
  if [ -z "$d" ]; then
    case $a in
      nemotron) pfx=nm_nemotron; script=scripts/run_nemotron_v1.sh; need=65000 ;;
      af3)      pfx=nm_af3;      script=scripts/run_af3_v1.sh;      need=30000 ;;
    esac
    cond=$b; order=$c; extra=""
  else
    pfx=$a; cond=$c; order=$d; need=30000; extra=""
    case $b in
      factorial) script=scripts/eval_stage2_factorial_v1.py ;;
      q3)        script=scripts/eval_stage2_q3_v1.py; need=45000 ;;
      new)       script=scripts/eval_newmodels_v1.py; extra="--model ${pfx#nm_}" ;;
    esac
  fi
  local tag=""; [ "$order" = swap ] && tag="_swap"
  local out="outputs/${pfx}_sound_${cond}${tag}_v1.jsonl"
  if [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ]; then echo "[skip] $(basename $out)" | tee -a "$LOG"; return; fi
  [ "$order" = swap ] && extra="$extra --swap_options"
  while :; do
    local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu")
    [ "$free" -ge "$need" ] && break
    sleep 45
  done
  echo "[gpu$gpu] $(date +%H:%M) start $pfx $cond $order" | tee -a "$LOG"
  if [[ $script == *.sh ]]; then
    CUDA_VISIBLE_DEVICES=$gpu bash "$script" --attr sound --cond "$cond" $extra >> "outputs/w7_${pfx}_${cond}${tag}.out" 2>&1 || true
  else
    CUDA_VISIBLE_DEVICES=$gpu python3 "$script" --attr sound --cond "$cond" $extra >> "outputs/w7_${pfx}_${cond}${tag}.out" 2>&1 || true
  fi
  echo "[gpu$gpu] $(date +%H:%M) done  $pfx $cond $order rows=$( [ -f "$out" ] && wc -l < "$out" || echo 0)" | tee -a "$LOG"
}
worker() { local g=$1 q=$2; while :; do local j; j=$(take "$q"); [ -z "$j" ] && break; run_job "$g" $j; done; echo "[gpu$g] empty" | tee -a "$LOG"; }
worker 0 "$QN" &
for g in 1 2 3; do worker "$g" "$QS" & done
wait
echo "[wave7] all done $(date)" | tee -a "$LOG"
