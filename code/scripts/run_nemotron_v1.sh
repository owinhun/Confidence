#!/bin/bash
# Nemotron 3 Nano Omni needs transformers 5.x (its remote code imports
# transformers.initialization, which does not exist in 4.57). The environment assembled for
# Audio Flamingo 3 under /data4/pyenv_af3 already provides it; injecting it here keeps the
# shared site-packages untouched, exactly as run_af3_v1.sh does.
set -u
cd /workspace/svhalluc/claude_track2
E=/data4/pyenv_af3
LD=$(python3 -c "import glob;print(':'.join(sorted(set(glob.glob('$E/nvidia/*/lib')))))")
export PYTHONPATH="$E" LD_LIBRARY_PATH="$LD:${LD_LIBRARY_PATH:-}"
export HF_HOME=/data4/hf_cache HF_HUB_CACHE=/data4/hf_cache/hub
export HF_MODULES_CACHE=/data4/hf_cache/modules_tf5
exec python3 scripts/eval_nemotron_v1.py "$@"
