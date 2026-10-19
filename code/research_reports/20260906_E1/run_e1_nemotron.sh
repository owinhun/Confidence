#!/bin/bash
# same environment injection as scripts/run_nemotron_v1.sh (transformers 5 via /data4/pyenv_af3), exec'ing the E1 wrapper
set -u
cd /workspace/svhalluc/claude_track2
E=/data4/pyenv_af3
LD=$(python3 -c "import glob;print(':'.join(sorted(set(glob.glob('$E/nvidia/*/lib')))))")
export PYTHONPATH="$E" LD_LIBRARY_PATH="$LD:${LD_LIBRARY_PATH:-}"
export HF_HOME=/data4/hf_cache HF_HUB_CACHE=/data4/hf_cache/hub HF_MODULES_CACHE=/data4/hf_cache/modules_tf5
exec python3 research_reports/20260906_E1/eval_e1_nemotron_v1.py "$@"
