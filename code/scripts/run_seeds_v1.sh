#!/bin/bash
# Seed-reproducibility for the evidence-use adapter (seeds 1 and 2 on GPUs 2/3).
set -u
CT2=/workspace/svhalluc/claude_track2
cd "$CT2"
S=/root/.claude/jobs/65f6dcfb/tmp/seeds_status.txt
: > "$S"
log(){ echo "[$(date -u +%H:%M)] $*" | tee -a "$S"; }
for pair in "1 2" "2 3"; do
  set -- $pair; seed=$1; gpu=$2
  ( log "seed $seed training on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python3 scripts/train_lora_evidence_v1.py \
      --data data/t3_evidence_train_v1.jsonl --out_dir outputs/lora_evidence_s${seed} \
      --lr 5e-5 --schedule constant --epochs 2 --val_limit 120 --seed $seed \
      > outputs/train_lora_evidence_s${seed}.log 2>&1
    log "seed $seed training done"
    CUDA_VISIBLE_DEVICES=$gpu python3 scripts/expert_inject_ta_t2_v1.py \
      --adapter outputs/lora_evidence_s${seed}/epoch2 \
      --out outputs/expert_inject_ta_s${seed}_v1.jsonl \
      > outputs/expert_inject_ta_s${seed}.log 2>&1
    log "seed $seed TA eval done"
    CUDA_VISIBLE_DEVICES=$gpu python3 scripts/eval_vggsparse_qwen_v1.py --mode inject \
      --adapter outputs/lora_evidence_s${seed}/epoch2 --suffix _s${seed} \
      > outputs/eval_vggs_s${seed}.log 2>&1
    log "seed $seed VGGS eval done" ) &
done
wait
log "SEEDS CHAIN TERMINATED"
