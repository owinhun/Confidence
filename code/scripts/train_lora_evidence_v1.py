#!/usr/bin/env python3
"""T3 evidence-use LoRA: expert report injected into the prompt.

LoRA SFT for audio-visual synchrony tuning on Qwen2.5-Omni-7B Thinker.

Task: 12s AV clip (use_audio_in_video=True) + SVHalluc-TA question
  "When the speech is heard, is the narrated event happening at the same
   time in the video? (A) Yes (B) No"
Label A = aligned, B = shifted.

Loss: cross-entropy on ONLY the single answer-letter token (A=32 / B=33,
verified at runtime) at the first generated position. We append the answer
token id to the chat-templated prompt's input_ids and set labels=-100
everywhere except that last position; the model's internal shift makes the
loss come from the logits at the final prompt position — identical to the
constrained-scoring measurement used at eval.

LoRA: language-model self-attention q/k/v/o projections ONLY (full-name
targeting; audio tower shares the q_proj/... suffixes and is excluded).
Vision/audio towers and all other weights frozen.

Run (GPU1):
  CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python claude_track2/scripts/train_lora_sync_v1.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "/workspace/svhalluc/src/qwen")

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from qwen_omni_utils import process_mm_info
from transformers import (Qwen2_5OmniProcessor,
                          Qwen2_5OmniThinkerForConditionalGeneration,
                          get_constant_schedule_with_warmup,
                          get_cosine_schedule_with_warmup)

ROOT = Path("/workspace/svhalluc")
MODEL_PATH = ROOT / "models/Qwen2.5-Omni-7B"
QUESTION = ("Is the audio track temporally aligned with the visual content of "
            "this video? (A) Yes (B) No")
_OLD_QUESTION = ("When the speech is heard, is the narrated event happening at the "
            "same time in the video? (A) Yes (B) No")
LETTER_IDS = {"A": 32, "B": 33}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str,
                   default=str(ROOT / "claude_track2/data/train_pairs_v2.jsonl"))
    p.add_argument("--fallback_data", type=str,
                   default=str(ROOT / "claude_track2/data/train_pairs_v1.jsonl"))
    p.add_argument("--out_dir", type=str,
                   default=str(ROOT / "claude_track2/outputs/lora_sync_v1"))
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--schedule", choices=["cosine", "constant"], default="cosine")
    p.add_argument("--warmup_steps", type=int, default=10)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=0,
                   help="if >0, stop after this many optimizer steps (smoke test)")
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--max_frames", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=10, help="log every N micro-steps")
    p.add_argument("--no_grad_ckpt", action="store_true",
                   help="disable gradient checkpointing (faster but OOMs on "
                        "long AV samples at 80GB — checkpointing is ON by default)")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--val_limit", type=int, default=0,
                   help="if >0, evaluate only the first N val rows (smoke test)")
    return p.parse_args()


def load_rows(args):
    path = Path(args.data)
    use = path
    if not path.exists() or sum(1 for _ in path.open()) < 100:
        print(f"[data] {path} absent or <100 rows -> falling back to {args.fallback_data}")
        use = Path(args.fallback_data)
    rows = [json.loads(l) for l in use.open() if l.strip()]
    train = [r for r in rows if r.get("split") == "train"]
    val = [r for r in rows if r.get("split") == "val"]
    print(f"[data] {use.name}: {len(rows)} rows -> train={len(train)} val={len(val)}")
    return train, val


def expert_report(s: float) -> str:
    return (
        "An audio-visual synchronization expert model analyzed this clip. "
        f"Estimated probability that the audio and the visual content are "
        f"time-aligned: {s:.2f}. Probabilities near 0 indicate the audio is "
        "out of sync with the video; higher values indicate alignment. Use "
        "this evidence together with the video and audio to answer.\n\n")


def build_messages(row, fps: int | float, max_frames: int):
    clip_path = str(ROOT / row["clip"])
    text = expert_report(row["s_near0"]) + QUESTION
    return [{"role": "user", "content": [
        {"type": "video", "video": clip_path, "fps": fps, "max_frames": max_frames},
        {"type": "text", "text": text},
    ]}]


def build_inputs(processor, messages, use_audio=True):
    text = processor.apply_chat_template(messages, add_generation_prompt=True,
                                         tokenize=False)
    audios, images, videos, video_kwargs = process_mm_info(
        messages, use_audio_in_video=use_audio, return_video_kwargs=True)
    if isinstance(video_kwargs.get("fps"), list) and len(video_kwargs["fps"]) == 1:
        video_kwargs["fps"] = video_kwargs["fps"][0]
    return processor(text=text, audio=audios, images=images, videos=videos,
                     return_tensors="pt", padding=True,
                     use_audio_in_video=use_audio, **video_kwargs)


def build_train_batch(processor, row, fps, max_frames):
    """Prompt inputs + answer letter appended; labels -100 except last pos."""
    inputs = build_inputs(processor, build_messages(row, fps, max_frames))
    ans_id = LETTER_IDS[row["label"]]
    ids = inputs["input_ids"]                                     # [1, L]
    inputs["input_ids"] = torch.cat(
        [ids, torch.tensor([[ans_id]], dtype=ids.dtype)], dim=1)
    am = inputs["attention_mask"]
    inputs["attention_mask"] = torch.cat(
        [am, torch.ones((1, 1), dtype=am.dtype)], dim=1)
    labels = torch.full_like(inputs["input_ids"], -100)
    labels[0, -1] = ans_id
    return inputs, labels


@torch.inference_mode()
def val_accuracy(model, processor, val_rows, fps, max_frames, device, dtype):
    """Constrained scoring on val split: softmax over A/B letter logits at the
    first generated position (no generation) — same measurement as eval."""
    model.eval()
    n_ok = n_correct = 0
    for row in val_rows:
        try:
            inputs = build_inputs(processor,
                                  build_messages(row, fps, max_frames))
            inputs = inputs.to(device).to(dtype)
            out = model(**inputs, use_audio_in_video=True, return_dict=True)
            logits = out.logits[0, -1, :].float()
            ab = torch.tensor([logits[LETTER_IDS["A"]].item(),
                               logits[LETTER_IDS["B"]].item()])
            probs = F.softmax(ab, dim=0)
            pred = "A" if probs[0] >= 0.5 else "B"
            n_ok += 1
            n_correct += int(pred == row["label"])
        except Exception as e:  # noqa: BLE001
            print(f"[val] SKIP {row['clip']}: {type(e).__name__}: {e}", flush=True)
    model.train()
    acc = n_correct / n_ok if n_ok else float("nan")
    return acc, n_correct, n_ok


def main():
    args = parse_args()
    if "expandable_segments" not in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
        print("[note] recommend PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    train_rows, val_rows = load_rows(args)

    print("[model] loading...", flush=True)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="flash_attention_2", local_files_only=True)
    processor = Qwen2_5OmniProcessor.from_pretrained(
        MODEL_PATH, local_files_only=True, use_fast=False)

    # Verify answer letters are single tokens with the expected ids.
    tok = processor.tokenizer
    for letter, expect in LETTER_IDS.items():
        enc = tok.encode(letter)
        assert enc == [expect], f"letter {letter}: expected [{expect}], got {enc}"
    print(f"[tok] letter ids verified: {LETTER_IDS}")

    # LoRA on language-model self-attn q/k/v/o ONLY (full names; the audio
    # tower shares the q_proj/k_proj/v_proj suffixes and must be excluded).
    pat = re.compile(r"^model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj$")
    targets = [n for n, _ in model.named_modules() if pat.fullmatch(n)]
    assert len(targets) == 4 * model.config.text_config.num_hidden_layers, \
        f"unexpected target count {len(targets)}"
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                      lora_dropout=args.lora_dropout, bias="none",
                      target_modules=targets, task_type=None)
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    if not args.no_grad_ckpt:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        print("[train] gradient checkpointing enabled")
    device, dtype = model.device, torch.bfloat16

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train_rows) / args.grad_accum)
    total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.epochs
    if args.schedule == "cosine":
        sched = get_cosine_schedule_with_warmup(opt, args.warmup_steps, total_steps)
    else:
        sched = get_constant_schedule_with_warmup(opt, args.warmup_steps)
    print(f"[train] epochs={args.epochs} grad_accum={args.grad_accum} "
          f"steps/epoch={steps_per_epoch} total_opt_steps={total_steps} "
          f"lr={args.lr} schedule={args.schedule}", flush=True)

    model.train()
    global_step = 0          # optimizer steps
    micro_in_accum = 0       # samples since last optimizer step
    running, running_n = 0.0, 0
    t0 = time.time()
    stop = False

    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train_rows)))
        random.shuffle(order)
        n_skipped = 0
        for micro_idx, ridx in enumerate(order):
            row = train_rows[ridx]
            try:
                inputs, labels = build_train_batch(processor, row,
                                                   args.fps, args.max_frames)
                inputs = inputs.to(device).to(dtype)
                labels = labels.to(device)
                out = model(**inputs, labels=labels, use_audio_in_video=True,
                            return_dict=True)
                loss = out.loss
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss {loss.item()}")
                (loss / args.grad_accum).backward()
                running += loss.item()
                running_n += 1
                micro_in_accum += 1
            except Exception as e:  # noqa: BLE001
                n_skipped += 1
                msg = str(e).splitlines()[0] if str(e) else ""
                print(f"[train] SKIP {row['clip']}: {type(e).__name__}: {msg}",
                      flush=True)
                if isinstance(e, torch.cuda.OutOfMemoryError):
                    opt.zero_grad(set_to_none=True)
                    micro_in_accum = 0
                    torch.cuda.empty_cache()
                continue

            if micro_in_accum >= args.grad_accum:
                torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1
                micro_in_accum = 0
                if args.max_steps > 0 and global_step >= args.max_steps:
                    stop = True

            if running_n and ((micro_idx + 1) % args.log_every == 0 or stop):
                mem = torch.cuda.max_memory_allocated() / 2**30
                print(f"[e{epoch} micro {micro_idx+1}/{len(order)} "
                      f"opt_step {global_step}] loss={running/running_n:.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e} mem={mem:.1f}GiB "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
                running, running_n = 0.0, 0
            if stop:
                break

        # flush a partial accumulation window at epoch end
        if micro_in_accum > 0 and not stop:
            torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            global_step += 1
            micro_in_accum = 0

        ep_dir = out_root / f"epoch{epoch}"
        model.save_pretrained(str(ep_dir))
        print(f"[save] adapter -> {ep_dir} (skipped {n_skipped} samples this epoch)",
              flush=True)

        if val_rows:
            if args.val_limit > 0:
                val_rows = val_rows[:args.val_limit]
            acc, c, n = val_accuracy(model, processor, val_rows,
                                     args.fps, args.max_frames, device, dtype)
            print(f"[val] epoch {epoch}: acc={c}/{n}={acc*100:.1f}%", flush=True)
        if stop:
            break

    print(f"[done] optimizer steps={global_step} wall={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
