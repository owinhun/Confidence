#!/bin/bash
set -x
cd /workspace/svhalluc
export CUDA_VISIBLE_DEVICES=1
P="python claude_track2/scripts/vllama_encoder_probe_v1.py"
$P --stage all 2>&1
$P --stage head --modality audio 2>&1
$P --stage head --modality visual 2>&1
$P --stage head --permute_labels 2>&1
$P --stage head --pair raw 2>&1
$P --stage head --pair raw --modality audio 2>&1
echo "ALL DONE"
