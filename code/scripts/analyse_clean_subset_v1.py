#!/usr/bin/env python3
"""Does the cascade result survive on clips whose labels a listener confirmed?

The reviews' one experiment-shaped objection is that the real cascade's ground truth is
VGGSound's sound-event labels, which we never re-listened to. Section 5's own CLAP check flags
three classes as weakly separated, but CLAP is also the detector, so that check cannot tell a
noisy label from a class CLAP is bad at.

A blind listening pass over those three classes (98 clips, the annotator never saw the manifest
label) settles it. Agreement is 94.7%: the two eating classes are 100%, and every disagreement
is in `people sneezing`. This recomputes the downstream policy deltas with the disagreements and
the unsure clips dropped, so the headline can be quoted on labels a human confirmed.

Dropping only 8 of 400 clips, so this is a sensitivity check rather than a re-estimate.
"""
from __future__ import annotations
import json, sys, os
sys.path.insert(0, "scripts")
import numpy as np
import analyse_cascade_v2 as C

CELLS = [("Qwen2.5", "s2"), ("Qwen3", "s2q3"), ("Qwen2-Audio", "nm_qwen2audio"),
         ("Nemotron", "nm_nemotron")]          # order-robust rows only; AF3 is sensitivity


def confirmed_clips():
    """Clips the listener either confirmed, or never had to judge."""
    man = {json.loads(l)["clip"]: json.loads(l)
           for l in open("data/held_attr_soundevent_v1.jsonl") if l.strip()}
    seen = {}
    for l in open("outputs/label_audit_v1.jsonl"):
        if l.strip():
            r = json.loads(l)
            seen[r["clip"]] = r["answer"]
    drop = set()
    for c, ans in seen.items():
        if ans == "unsure" or (ans == "yes") != (man[c]["label"] == "A"):
            drop.add(c)
    return set(man) - drop, drop


def audit_counts():
    """Audit tallies, computed rather than typed.

    AuditN and AuditAgree were string literals here for several drafts. They happened to be
    right, but nothing checked them, and the same pattern hid a wrong claim elsewhere in the
    paper. They are derived now. Note the two denominators differ: agreement is over the clips
    the listener could actually judge, so the unsure ones are excluded from it and enter the
    drop count instead.
    """
    man = {json.loads(l)["clip"]: json.loads(l)
           for l in open("data/held_attr_soundevent_v1.jsonl") if l.strip()}
    rows = [json.loads(l) for l in open("outputs/label_audit_v1.jsonl") if l.strip()]
    scored = [r for r in rows if r["answer"] != "unsure"]
    agree = sum((r["answer"] == "yes") == (man[r["clip"]]["label"] == "A") for r in scored)
    # how many of the judged clips asked whether the queried class is *absent* -- the half a
    # single-label dataset cannot vouch for on its own, and the half reviews ask about
    neg = sum(man[r["clip"]]["label"] == "B" for r in scored)
    uns = len(rows) - len(scored)
    return len(rows), len(scored), agree, neg, uns


def paired_on(pfx, a, b, keep):
    """Order-averaged a-minus-b restricted to the kept clips, source-video bootstrap."""
    per, pts = [], []
    for sfx in ("", "_swap"):
        A, B = C.L(f"{pfx}_sound_{a}{sfx}"), C.L(f"{pfx}_sound_{b}{sfx}")
        if A is None or B is None:
            continue
        ks = sorted((set(A) & set(B)) & keep)
        d = np.array([A[k]["correct"] - B[k]["correct"] for k in ks], dtype=float)
        pts.append(d.mean() * 100)
        per.append((d, ks))
    if not per:
        return None
    vids = sorted({C.video(k) for _, ks in per for k in ks})
    vi = {v: i for i, v in enumerate(vids)}
    bk = [[np.where(np.array([vi[C.video(k)] for k in ks]) == i)[0] for i in range(len(vids))]
          for _, ks in per]
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(2000):
        sel = rng.integers(0, len(vids), len(vids))
        vals = []
        for (d, _), b_ in zip(per, bk):
            idx = np.concatenate([b_[i] for i in sel if b_[i].size])
            if idx.size:
                vals.append(d[idx].mean() * 100)
        if vals:
            bs.append(float(np.mean(vals)))
    return (float(np.mean(pts)), float(np.percentile(bs, 2.5)),
            float(np.percentile(bs, 97.5)), len(per[0][1]))


if __name__ == "__main__":
    keep, drop = confirmed_clips()
    print(f"청취 확인 {len(keep)}클립, 제외 {len(drop)}클립 (불일치+모름)\n")
    print(f"{'셀':12} {'전체 400':>26} {'확인된 클립만':>26}")
    print("-" * 68)
    out = []
    for nm, pfx in CELLS:
        full = C.paired(pfx, "cascadex_plain", "none")
        cl = paired_on(pfx, "cascadex_plain", "none", keep)
        f = lambda r: "--" if r is None else f"{r[0]:+6.1f} [{r[1]:+5.1f},{r[2]:+5.1f}]"
        print(f"{nm:12} {f(full):>26} {f(cl):>26}")
        if cl:
            out.append((nm, cl))
    if out:
        v = sorted(r[0] for _, r in out)
        neg = [n for n, r in out if r[2] < 0]
        print(f"\n확인된 클립만: 손실 {-v[-1]:.1f}--{-v[0]:.1f}점, CI가 0 아래 {len(neg)}/{len(out)}셀")
        import pathlib
        _an, _asc, _aag, _aneg, _auns = audit_counts()
        pathlib.Path("paper/auto_clean.tex").write_text(
            "% Generated by scripts/analyse_clean_subset_v1.py -- do not edit by hand.\n"
            f"\\newcommand{{\\AuditN}}{{{_an}}}\n"
            f"\\newcommand{{\\AuditScoredN}}{{{_asc}}}\n"
            f"\\newcommand{{\\AuditNegN}}{{{_aneg}}}\n"
            f"\\newcommand{{\\AuditAgree}}{{{100*_aag/_asc:.1f}}}\n"
            f"\\newcommand{{\\AuditDropN}}{{{len(drop)}}}\n"
            f"\\newcommand{{\\AuditDisN}}{{{_asc - _aag}}}\n"
            f"\\newcommand{{\\AuditUnsN}}{{{_auns}}}\n"
            f"\\newcommand{{\\CleanMin}}{{{-v[-1]:.1f}}}\n"
            f"\\newcommand{{\\CleanMax}}{{{-v[0]:.1f}}}\n"
            f"\\newcommand{{\\CleanNegN}}{{{len(neg)}}}\n"
            f"\\newcommand{{\\CleanCells}}{{{len(out)}}}\n")
        print("paper/auto_clean.tex 작성")
