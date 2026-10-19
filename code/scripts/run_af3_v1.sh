#!/bin/bash
# AF3 needs transformers 5.x / torch 2.5, which the shared environment cannot provide without
# breaking the other five models. Everything it needs is assembled under /data4/pyenv_af3 and
# injected here; the shared site-packages stay untouched.
set -u
cd /workspace/svhalluc/claude_track2
E=/data4/pyenv_af3
LD=$(python3 -c "import glob;print(':'.join(sorted(set(glob.glob('$E/nvidia/*/lib')))))")
export PYTHONPATH="$E" LD_LIBRARY_PATH="$LD:${LD_LIBRARY_PATH:-}"
export HF_HOME=/data4/hf_cache HF_HUB_CACHE=/data4/hf_cache/hub
exec python3 scripts/eval_af3_v1.py "$@"
