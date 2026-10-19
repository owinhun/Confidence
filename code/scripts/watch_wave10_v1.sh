#!/usr/bin/env bash
# Requeue wave10 jobs that died with rows=0.
#
# The runner gates on nvidia-smi free memory but reserves nothing, so two workers can read the
# same free GPU a moment apart and both launch onto it. The larger model then OOMs at load. The
# job is not broken -- it just lost a race -- so the fix is to put it back rather than to debug
# it. Runs until the wave reports done, and never requeues the same job twice in a row.
set -uo pipefail
cd /workspace/svhalluc/claude_track2
Q=/root/.claude/jobs/65f6dcfb/tmp/w10.txt
LOCK=/root/.claude/jobs/65f6dcfb/tmp/w10.lock
LOG=outputs/wave10_log.txt
SEEN=/root/.claude/jobs/65f6dcfb/tmp/w10_requeued.txt
: > "$SEEN"

# map a "[gpuN] HH:MM done  pfx attr cond five rows=0" line back to its queue line
requeue() {
  local pfx=$1 attr=$2 cond=$3 five=$4 kind
  case $pfx in
    s2)  kind=fact ;;  s2q3) kind=q3 ;;  s2vl) kind=vl ;;
    nm_qwen2audio) kind=new ;;  nm_af3) kind=af3 ;;  nm_nemotron) kind=nemo ;;
    cs_qwen25|cs_qwen3) kind=cs; pfx=${pfx#cs_} ;;
    *) return ;;
  esac
  echo "$pfx $kind $attr $cond $five"
}

while ! grep -q "wave10. all done" "$LOG" 2>/dev/null; do
  while read -r line; do
    [[ $line =~ rows=0$ ]] || continue
    set -- $line
    # [gpuN] HH:MM done pfx attr cond five rows=0
    local_job=$(requeue "$4" "$5" "$6" "$7")
    [ -z "$local_job" ] && continue
    grep -qxF "$local_job" "$SEEN" && continue
    echo "$local_job" >> "$SEEN"
    flock "$LOCK" bash -c "echo '$local_job' >> $Q"
    echo "[watch] requeued: $local_job" | tee -a "$LOG"
  done < <(grep " done " "$LOG" | grep "rows=0$")
  sleep 300
done
echo "[watch] wave10 finished, exiting" | tee -a "$LOG"
