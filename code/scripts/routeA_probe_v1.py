#!/usr/bin/env python3
"""Route A scout: shift-profile probing. For each TA item (both classes),
build 3 clips with IDENTICAL video (v[4:T-4]) and audio = aligned [4:T-4],
lead [8:T], lag [0:T-8]. Predict Yes iff p_yes(aligned) > max(p_yes(shifted)).
Under no-signal null, P(argmax at aligned) = 1/3 -> Yes-rate ~33% both classes;
signal shows as class separation. n=60 mixed."""
import json, subprocess, sys
from pathlib import Path
sys.path.insert(0, "/workspace/svhalluc/src/qwen")
import torch, torch.nn.functional as F
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
ROOT = Path("/workspace/svhalluc")
D = 4.0
CLIPS = ROOT / "claude_track2/outputs/routeA_clips"; CLIPS.mkdir(exist_ok=True)
OUT = ROOT / "claude_track2/outputs/routeA_probe_v1.jsonl"
rows = [json.loads(l) for l in open(ROOT/"data/ter_large_eval_v1.jsonl") if l.strip()]
ta = [r for r in rows if r["task"]=="TA"]
def dur(p):
    o = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0",p],capture_output=True,text=True)
    try: return float(o.stdout.strip())
    except: return None
def cut(src,dst,v0,v1,a0,a1):
    r = subprocess.run(["ffmpeg","-y","-v","error","-i",src,"-filter_complex",
        f"[0:v]trim={v0}:{v1},setpts=PTS-STARTPTS[v];[0:a]atrim={a0}:{a1},asetpts=PTS-STARTPTS[a]",
        "-map","[v]","-map","[a]","-c:v","libx264","-preset","veryfast","-crf","23","-c:a","aac",str(dst)],capture_output=True)
    return r.returncode==0 and dst.stat().st_size>0
model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(ROOT/"models/Qwen2.5-Omni-7B",
    dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="flash_attention_2", local_files_only=True)
model.eval()
proc = Qwen2_5OmniProcessor.from_pretrained(ROOT/"models/Qwen2.5-Omni-7B", local_files_only=True, use_fast=False)
def p_yes(clip, q):
    msgs=[{"role":"user","content":[{"type":"video","video":str(clip),"fps":2.0,"max_frames":48},{"type":"text","text":q}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    a,i,v,vk = process_mm_info(msgs, use_audio_in_video=True, return_video_kwargs=True)
    if isinstance(vk.get("fps"),list) and len(vk["fps"])==1: vk["fps"]=vk["fps"][0]
    ins = proc(text=text,audio=a,images=i,videos=v,return_tensors="pt",padding=True,use_audio_in_video=True,**vk).to(model.device).to(model.dtype)
    with torch.inference_mode():
        out = model(**ins, return_dict=True, use_audio_in_video=True)
    lg = out.logits[0,-1,:].float()
    ab = torch.tensor([lg[32].item(), lg[33].item()])
    return F.softmax(ab,dim=0)[0].item()
done=set()
if OUT.exists():
    done={json.loads(l)["sample_id"] for l in open(OUT) if l.strip()}
n_taken=0
with OUT.open("a") as f:
    for r in ta:
        if n_taken>=60: break
        sid=r["sample_id"]
        T=dur(r["video_path"])
        if T is None or T<16: continue
        n_taken+=1
        if sid in done: continue
        try:
            v0,v1=D,T-D
            variants={"aligned":(D,T-D),"lead":(2*D,T),"lag":(0.0,T-2*D)}
            ps={}
            for k,(a0,a1) in variants.items():
                c=CLIPS/f"{sid}_{k}.mp4"
                if not c.exists() and not cut(r["video_path"],c,v0,v1,a0,a1): raise RuntimeError("cut")
                ps[k]=p_yes(c,r["question"])
            pred="A" if ps["aligned"]>max(ps["lead"],ps["lag"]) else "B"
            f.write(json.dumps({"sample_id":sid,"answer":r["answer"],**{f"p_{k}":v for k,v in ps.items()},
                                "prediction":pred,"correct":pred==r["answer"]})+"\n"); f.flush()
            print(sid,pred,r["answer"],flush=True)
        except Exception as e:
            print(sid,"ERROR",e,flush=True)
jr=[json.loads(l) for l in open(OUT) if l.strip()]
n=len(jr); c=sum(r["correct"] for r in jr)
print(f"FINAL n={n} acc={c/n*100:.1f}%")
