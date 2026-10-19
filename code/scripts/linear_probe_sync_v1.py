#!/usr/bin/env python3
"""Representational-sufficiency diagnostic: can a linear probe on FROZEN hidden
states separate aligned vs shifted clips? ytid-grouped CV (no video leakage).
If ~50%: info absent -> no adapter can work; honest stop. If >>50%: info
present -> extended-target LoRA (run 3) justified."""
import json, sys
from pathlib import Path
sys.path.insert(0, "/workspace/svhalluc/src/qwen")
import numpy as np, torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
ROOT = Path("/workspace/svhalluc")
LAYERS = [8, 14, 20, 27]
rows = [json.loads(l) for l in open(ROOT/"claude_track2/data/train_pairs_easy_v1.jsonl") if l.strip()]
Q = "When the speech is heard, is the narrated event happening at the same time in the video? (A) Yes (B) No"
model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(ROOT/"models/Qwen2.5-Omni-7B",
    dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="flash_attention_2", local_files_only=True)
model.eval()
proc = Qwen2_5OmniProcessor.from_pretrained(ROOT/"models/Qwen2.5-Omni-7B", local_files_only=True, use_fast=False)
feats, labels, groups = [], [], []
outp = ROOT/"claude_track2/outputs/probe_feats_v1.npz"
for k, r in enumerate(rows):
    try:
        msgs=[{"role":"user","content":[{"type":"video","video":str(ROOT/r["clip"]),"fps":2.0,"max_frames":24},{"type":"text","text":Q}]}]
        text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        a,i,v,vk = process_mm_info(msgs, use_audio_in_video=True, return_video_kwargs=True)
        if isinstance(vk.get("fps"),list) and len(vk["fps"])==1: vk["fps"]=vk["fps"][0]
        ins = proc(text=text,audio=a,images=i,videos=v,return_tensors="pt",padding=True,use_audio_in_video=True,**vk).to(model.device).to(model.dtype)
        with torch.inference_mode():
            out = model(**ins, return_dict=True, use_audio_in_video=True, output_hidden_states=True)
        f = np.concatenate([out.hidden_states[L][0,-1,:].float().cpu().numpy() for L in LAYERS])
        feats.append(f); labels.append(1 if r["label"]=="A" else 0); groups.append(r["ytid"])
        if k % 25 == 0: print(k, flush=True)
    except Exception as e:
        print(k, "ERR", e, flush=True)
np.savez(outp, X=np.array(feats), y=np.array(labels), g=np.array(groups))
X, y, g = np.array(feats), np.array(labels), np.array(groups)
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
H = X.shape[1] // len(LAYERS)
for li, L in enumerate(LAYERS):
    Xi = X[:, li*H:(li+1)*H]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
    sc = cross_val_score(clf, Xi, y, cv=GroupKFold(5), groups=g)
    print(f"layer {L}: probe acc={sc.mean()*100:.1f}% (+-{sc.std()*100:.1f}) n={len(y)}")
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
sc = cross_val_score(clf, X, y, cv=GroupKFold(5), groups=g)
print(f"ALL layers concat: probe acc={sc.mean()*100:.1f}% (+-{sc.std()*100:.1f})")
