#!/bin/bash
# T3 chain: wait for prerequisites, score train pairs, merge, train evidence-use
# LoRA, then eval the adapter on both benchmarks.
set -u
CT2=/workspace/svhalluc/claude_track2
cd "$CT2"

# 1. wait for the T3 pair build (CPU job)
while pgrep -f build_vggtrain_pairs_v1.py > /dev/null; do sleep 120; done
echo "[t3] pair build done: $(wc -l < data/vggtrain_pairs_v1.jsonl) clips"

# 2. wait for GPU0 to be free of the vggsparse scorer, then score train pairs on GPU0
while pgrep -f synchformer_vggsparse_v1.py > /dev/null; do sleep 120; done
CUDA_VISIBLE_DEVICES=0 python3 scripts/synchformer_vggtrain_v1.py \
  > outputs/synchformer_vggtrain_v1.log 2>&1
echo "[t3] train-pair scoring done: $(wc -l < outputs/synchformer_vggtrain_v1.jsonl)"

# 3. merge manifest + expert scores -> training jsonl
python3 - << 'PYEOF'
import json
from pathlib import Path
CT2 = Path("/workspace/svhalluc/claude_track2")
scores = {json.loads(l)["clip"]: json.loads(l)["s_near0"]
          for l in (CT2 / "outputs/synchformer_vggtrain_v1.jsonl").open() if l.strip()}
out = (CT2 / "data/t3_evidence_train_v1.jsonl").open("w")
n = 0
for l in (CT2 / "data/vggtrain_pairs_v1.jsonl").open():
    if not l.strip():
        continue
    r = json.loads(l)
    if r["clip"] in scores:
        r["s_near0"] = scores[r["clip"]]
        out.write(json.dumps(r) + "\n")
        n += 1
out.close()
print(f"[t3] merged {n} rows -> t3_evidence_train_v1.jsonl")
PYEOF

# 4. train evidence-use LoRA on GPU0 (registered recipe: r=16/a=32, lr 5e-5 const, 2 epochs)
CUDA_VISIBLE_DEVICES=0 python3 scripts/train_lora_evidence_v1.py \
  --data data/t3_evidence_train_v1.jsonl \
  --out_dir outputs/lora_evidence_v1 \
  --lr 5e-5 --schedule constant --epochs 2 --val_limit 120 \
  > outputs/train_lora_evidence_v1.log 2>&1
echo "[t3] training done"

# 5. pick last epoch adapter
ADAPTER=$(ls -d outputs/lora_evidence_v1/epoch* 2>/dev/null | sort -V | tail -1)
echo "[t3] adapter: $ADAPTER"
if [ -z "$ADAPTER" ]; then echo "[t3] NO ADAPTER — abort evals"; exit 1; fi

# 6. wait for GPU1 to be free of the frozen-model evals, then eval adapter
while pgrep -f eval_vggsparse_qwen_v1.py > /dev/null; do sleep 120; done
CUDA_VISIBLE_DEVICES=1 python3 scripts/eval_vggsparse_qwen_v1.py --mode inject \
  --adapter "$ADAPTER" --suffix _t3adapter \
  > outputs/eval_vggsparse_qwen_inject_t3adapter_v1.log 2>&1
echo "[t3] vggsparse adapter eval done"
CUDA_VISIBLE_DEVICES=1 python3 scripts/expert_inject_ta_t2_v1.py \
  --adapter "$ADAPTER" --out outputs/expert_inject_ta_t3adapter_v1.jsonl \
  > outputs/expert_inject_ta_t3adapter_v1.log 2>&1
echo "[t3] TA adapter eval done — T3 CHAIN ALL DONE"
