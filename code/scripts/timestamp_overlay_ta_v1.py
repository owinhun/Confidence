#!/usr/bin/env python3
"""Symbolic-injection variant: burn a running clock into frames (drawtext pts)
and state the speech interval (frozen VAD cues) in the question, so synchrony
becomes a reasoning task over capacities the model HAS (OCR + semantic match).
n=60 mixed TA items, constrained scoring, original layout."""
import json, subprocess, sys
from pathlib import Path
sys.path.insert(0, "/workspace/svhalluc/src/qwen")
import torch, torch.nn.functional as F
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
ROOT = Path("/workspace/svhalluc")
CLIPS = ROOT/"claude_track2/outputs/ts_overlay_clips"; CLIPS.mkdir(exist_ok=True)
OUT = ROOT/"claude_track2/outputs/timestamp_overlay_ta_v1.jsonl"
rows = [json.loads(l) for l in open(ROOT/"data/ter_large_eval_v1.jsonl") if l.strip()]
ta = [r for r in rows if r["task"]=="TA"]
cues = {json.loads(l)["video_id"]: json.loads(l) for l in open(ROOT/"outputs/ter_cues_large_v1.jsonl") if l.strip()}
def overlay(src,dst):
    if dst.exists() and dst.stat().st_size>0: return True
    r = subprocess.run(["ffmpeg","-y","-v","error","-i",src,"-vf",
        "drawtext=text='%{pts\\:hms}':x=10:y=10:fontsize=28:fontcolor=yellow:box=1:boxcolor=black@0.6",
        "-c:v","libx264","-preset","veryfast","-crf","23","-c:a","copy",str(dst)],capture_output=True)
    return r.returncode==0 and dst.stat().st_size>0
model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(ROOT/"models/Qwen2.5-Omni-7B",
    dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="flash_attention_2", local_files_only=True)
model.eval()
proc = Qwen2_5OmniProcessor.from_pretrained(ROOT/"models/Qwen2.5-Omni-7B", local_files_only=True, use_fast=False)
def p_yes(clip,q):
    msgs=[{"role":"user","content":[{"type":"video","video":str(clip),"fps":2.0,"max_frames":64},{"type":"text","text":q}]}]
    text=proc.apply_chat_template(msgs,add_generation_prompt=True,tokenize=False)
    a,i,v,vk=process_mm_info(msgs,use_audio_in_video=True,return_video_kwargs=True)
    if isinstance(vk.get("fps"),list) and len(vk["fps"])==1: vk["fps"]=vk["fps"][0]
    ins=proc(text=text,audio=a,images=i,videos=v,return_tensors="pt",padding=True,use_audio_in_video=True,**vk).to(model.device).to(model.dtype)
    with torch.inference_mode():
        out=model(**ins,return_dict=True,use_audio_in_video=True)
    lg=out.logits[0,-1,:].float()
    return F.softmax(torch.tensor([lg[32].item(),lg[33].item()]),dim=0)[0].item()
done=set()
if OUT.exists(): done={json.loads(l)["sample_id"] for l in open(OUT) if l.strip()}
n=0
with OUT.open("a") as f:
    for r in ta:
        if n>=60: break
        cue = cues.get(r["video_id"])
        if not cue or cue.get("vad_status")!="ok": continue
        n+=1
        sid=r["sample_id"]
        if sid in done: continue
        try:
            dst=CLIPS/f"{sid}.mp4"
            if not overlay(r["video_path"],dst): raise RuntimeError("overlay")
            s,e=cue["speech_start_sec"],cue["speech_end_sec"]
            asr=(cue.get("asr_transcript") or "")[:200]
            q=(f"A running clock (HH:MM:SS.ss) is burned into the top-left of each frame. "
               f"The speech occurs from {s:.1f}s to {e:.1f}s and says: \"{asr}\". "
               f"Check what is visible during frames whose clock reads {s:.1f}-{e:.1f} seconds. "
               + r["question"])
            for variant,qq,ans in (("overlay",q,r["answer"]),):
                p=p_yes(dst,qq)
                pred="A" if p>=0.5 else "B"
                f.write(json.dumps({"sample_id":sid,"variant":variant,"answer":ans,"p_yes":p,
                                    "prediction":pred,"correct":pred==ans})+"\n"); f.flush()
                print(sid,f"p={p:.3f} pred={pred} ans={ans}",flush=True)
        except Exception as ex:
            print(sid,"ERR",ex,flush=True)
jr=[json.loads(l) for l in open(OUT) if l.strip()]
nn=len(jr); c=sum(r["correct"] for r in jr)
from collections import Counter
print(f"FINAL n={nn} acc={c}/{nn}={c/nn*100:.1f}% dist={Counter(r['prediction'] for r in jr)}")
