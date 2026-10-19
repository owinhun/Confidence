#!/usr/bin/env python3
"""Every number the body quotes, computed here and emitted as a LaTeX macro.

Nine numerical errors reached a built PDF in one day, in four flavours: tables typed by hand,
prose left stale after the data behind it changed, two different quantities quoted in adjacent
sentences as though they were one, and a range quoted from a subset of the cells it claimed to
cover. Careful reading caught them one at a time and missed others; the only fix that scales is
to leave no place in the body where a number can be typed.

This file is that place. Each block below computes one group of quoted figures and writes it to
`paper/auto_all.tex`. A number that is not here does not belong in the body.

Definitional constants (the gate bounds, the numeric ladder's endpoints) stay literal in the
text: they define the protocol rather than report a measurement, and a generator that emitted
them would be pretending they came from data.
"""

from __future__ import annotations

import json
import os
import pathlib

import numpy as np

EPS, BOOTS = 1e-6, 2000
OUT = "outputs"
M: dict[str, str] = {}


def L(name):
    p = f"{OUT}/{name}_v1.jsonl"
    if not os.path.exists(p):
        return None
    d = {}
    for line in open(p):
        if line.strip():
            r = json.loads(line)
            d[r["clip"]] = r
    return d if len(d) >= 400 else None


def marg(r, swapped=False):
    pa = min(max(r["p_A"], EPS), 1 - EPS)
    v = (np.log(pa) - np.log(1 - pa)) * (1 if r["label"] == "A" else -1)
    return -v if swapped else v


def vid(c):
    return c.split("/")[-1].split("_")[0]


def put(k, v, nd=2):
    M[k] = v if isinstance(v, str) else f"{v:.{nd}f}"


def rng_put(prefix, xs, nd=2):
    xs = [x for x in xs if x is not None]
    put(prefix + "Min", min(xs), nd)
    put(prefix + "Max", max(xs), nd)


LAD = ["v_vlow", "v_low", "v_mod", "v_high", "v_vhigh"]
NEMO = [("v_vlow", "v_vlow_T"), ("anon_low", "true_low"), ("v_mod", "v_mod_T"),
        ("anon_high", "true_high_T"), ("v_vhigh", "v_vhigh_T")]
CELLS = [("Qwen2.5", "s2", "cs_qwen25", "sound", None),
         ("Qwen3", "s2q3", "cs_qwen3", "sound", None),
         ("Qwen3/noise", "s2q3", "cs_qwen3", "noise", None),
         ("VideoLLaMA", "s2vl", "cs_vllama", "colour", None),
         ("Qwen2-Audio", "nm_qwen2audio", "nm_qwen2audio", "sound", None),
         ("AF3", "nm_af3", "nm_af3", "sound", None),
         ("Nemotron/snd", "nm_nemotron", "nm_nemotron", "sound", NEMO),
         ("Nemotron/col", "nm_nemotron", "nm_nemotron", "colour", NEMO)]


def ladder():
    """beta at each declared rung, per cell, on unaided-correct items."""
    out = {}
    for name, pfx, lad, attr, nemo in CELLS:
        b = L(f"{pfx}_{attr}_none")
        if b is None:
            continue
        C = [k for k in b if b[k]["correct"]]
        ys = []
        for i in range(5):
            fs, ts = ((f"{pfx}_{attr}_{nemo[i][0]}", f"{pfx}_{attr}_{nemo[i][1]}") if nemo
                      else (f"{lad}_{attr}_{LAD[i]}", f"{lad}_{attr}_{LAD[i]}_T"))
            F, T = L(fs), L(ts)
            if F is None or T is None:
                ys.append(None)
                continue
            ks = [k for k in C if k in F and k in T]
            ys.append(float(np.mean([0.5 * (marg(T[k]) - marg(F[k])) for k in ks])))
        out[name] = ys
    return out


if __name__ == "__main__":
    lad = ladder()

    # --- Sec. 5: the ladder's top rung, and the middle rung the calibration argument uses
    tops = [v[4] for v in lad.values() if v[4] is not None]
    mids = [v[2] for v in lad.values() if v[2] is not None]
    rng_put("BetaTop", tops)
    rng_put("BetaMid", mids)
    put("BetaCells", str(len(lad)))
    def _mono(v):
        xs = [x for x in v if x is not None]
        return all(a <= b for a, b in zip(xs, xs[1:]))
    mono = sum(1 for v in lad.values() if _mono(v))
    put("BetaMonoN", str(mono))
    # AF3 clears the gate in one answer order only, so it is a sensitivity cell everywhere else
    # in the paper. Quoting "7 of 7" here folded it back into a primary claim and contradicted
    # the abstract's "six cells, with a seventh (AF3)". Emit the primary-only counts too.
    _prim = {k: v for k, v in lad.items() if "AF3" not in k}
    put("BetaPrimN", str(len(_prim)))
    put("BetaMonoPrimN", str(sum(1 for v in _prim.values() if _mono(v))))
    put("BetaAFThreeMono", "yes" if _mono(lad["AF3"]) else "no")

    # --- Sec. 5: an unaided-correct false claim at very low can *raise* accuracy
    b = L("nm_nemotron_colour_none")
    F = L("nm_nemotron_colour_v_vlow")
    if b and F:
        ks = sorted(set(b) & set(F))
        put("NemoColGain",
            (np.mean([F[k]["correct"] for k in ks]) - np.mean([b[k]["correct"] for k in ks])) * 100,
            1)

    # --- Sec. 5: accuracy looks flat on AF3 while the margin moves
    b = L("nm_af3_sound_none")
    C = [k for k in b if b[k]["correct"]]
    accs, mars = [], []
    for lv in LAD:
        F = L(f"nm_af3_sound_{lv}")
        if F is None:
            continue
        ks = [k for k in C if k in F]
        accs.append(np.mean([F[k]["correct"] for k in ks]) * 100)
        mars.append(np.mean([marg(F[k]) for k in ks]))
    rng_put("AFThreeAcc", accs, 1)
    put("AFThreeMargHi", mars[0])
    put("AFThreeMargLo", mars[-1])

    # --- Sec. 5: the graded-noise cell where override saturates but the margin does not
    b = L("s2_grad_noise_g4_none")
    if b:
        C = [k for k in b if b[k]["correct"]]
        put("GradNoiseC", str(len(C)))
        for tag, hi, lo in (("Conf", "anon_high", "anon_low"),
                            ("Prio", "priority_high", "priority_low")):
            A, B = L(f"s2_grad_noise_g4_{hi}"), L(f"s2_grad_noise_g4_{lo}")
            if A and B:
                ks = [k for k in C if k in A and k in B]
                # the paper's override metric is macro-POR, averaged over the two claim
                # directions; the raw flip rate is a different number and quoting it here once
                # put 43.8 in the text where the metric is 50.0
                _per = [[A[k]["pred"] != A[k]["label"] for k in ks if b[k]["label"] == lb]
                        for lb in ("A", "B")]
                put(f"GradNoiseOR{tag}",
                    float(np.mean([np.mean(v) for v in _per if v])) * 100, 1)
                put(f"GradNoiseFlip{tag}",
                    float(np.mean([A[k]["pred"] != A[k]["label"] for k in ks])) * 100, 1)
                put(f"GradNoiseMarg{tag}",
                    np.mean([marg(A[k]) for k in ks]) - np.mean([marg(B[k]) for k in ks]))

    # --- Sec. 5: alpha reverses sign on the complement
    flips, tot, example = 0, 0, None
    prim_flips, prim_tot = 0, 0
    for name, pfx, _lad, attr, nemo in CELLS:
        b = L(f"{pfx}_{attr}_none")
        T = L(f"{pfx}_{attr}_true_high") or L(f"{pfx}_{attr}_true_high_T")
        F = L(f"{pfx}_{attr}_anon_high")
        if not all([b, T, F]):
            continue
        vals = []
        for want in (True, False):
            ks = [k for k in b if bool(b[k]["correct"]) == want and k in T and k in F]
            if len(ks) < 12:
                vals.append(None)
                continue
            vals.append(float(np.mean([0.5 * ((marg(T[k]) - marg(b[k]))
                                              + (marg(F[k]) - marg(b[k]))) for k in ks])))
        if None in vals:
            continue
        tot += 1
        _f = (vals[0] > 0) != (vals[1] > 0)
        if _f:
            flips += 1
            if name == "Qwen2-Audio":
                example = vals
        # AF3 is the order-sensitive cell reported separately; the alpha count must be quotable
        # on the primary set alone, like every other count in the paper
        if "AF3" not in name:
            prim_tot += 1
            prim_flips += int(_f)
    put("AlphaFlipN", str(flips))
    put("AlphaTotN", str(tot))
    put("AlphaPrimFlipN", str(prim_flips))
    put("AlphaPrimTotN", str(prim_tot))
    if example:
        put("AlphaExHi", example[0])
        put("AlphaExLo", example[1])

    # --- Sec. 6: the signal-masked control
    masked = []
    for name, pfx, attr in (("AF3", "nm_af3", "sound"), ("Nemotron/snd", "nm_nemotron", "sound"),
                            ("Nemotron/col", "nm_nemotron", "colour"),
                            ("Qwen2-Audio", "nm_qwen2audio", "sound")):
        b = L(f"{pfx}_{attr}_none")
        T = L(f"{pfx}_{attr}_true_high") or L(f"{pfx}_{attr}_true_high_T")
        F = L(f"{pfx}_{attr}_anon_high")
        Tm, Fm = L(f"{pfx}_{attr}_true_high_masked"), L(f"{pfx}_{attr}_anon_high_masked")
        if not all([b, T, F, Tm, Fm]):
            continue
        C = [k for k in b if b[k]["correct"]]
        ks = [k for k in C if all(k in x for x in (T, F, Tm, Fm))]
        masked.append((float(np.mean([0.5 * (marg(T[k]) - marg(F[k])) for k in ks])),
                       float(np.mean([0.5 * (marg(Tm[k]) - marg(Fm[k])) for k in ks]))))
    put("MaskRiseN", str(sum(1 for a, m in masked if m > a)))
    put("MaskTotN", str(len(masked)))
    put("MaskRiseSpan", "; ".join(f"${a:+.2f}\\to{m:+.2f}$" for a, m in masked if m > a))
    fall = [(a, m) for a, m in masked if m <= a]
    if fall:
        put("MaskFall", f"${fall[0][0]:+.2f}\\to{fall[0][1]:+.2f}$")


    # --- Sec. 3: the CLAP cross-check on query-class labels
    CLS = {json.loads(l)["clip"]: json.loads(l)["query_class"]
           for l in open("data/held_attr_soundevent_v1.jsonl") if l.strip()}
    cal = {json.loads(l)["clip"]: json.loads(l) for l in open(f"{OUT}/clap_cascade_v1.jsonl")}
    WEAK = {"people eating apple", "people eating crisps", "people sneezing"}
    seps = {}
    for cls in set(CLS.values()):
        a = [cal[c]["p_cal"] for c, q in CLS.items() if q == cls and cal[c]["label"] == "A"]
        b_ = [cal[c]["p_cal"] for c, q in CLS.items() if q == cls and cal[c]["label"] == "B"]
        if a and b_:
            seps[cls] = np.mean(a) - np.mean(b_)
    strong = [v for k, v in seps.items() if k not in WEAK]
    weak = [v for k, v in seps.items() if k in WEAK]
    rng_put("LabelStrong", strong)
    rng_put("LabelWeak", weak)
    put("LabelStrongN", str(len(strong)))
    put("LabelWeakN", str(len(weak)))
    put("LabelClassN", str(len(seps)))

    # --- Sec. 4: the field-name increment at `low`, on the false-claim margin
    lows = []
    low_names = []
    for name, pfx, _l, attr, _n in CELLS:
        vals = []
        for tag in ("", "_swap"):
            b_ = L(f"{pfx}_{attr}_none{tag}")
            A, P = L(f"{pfx}_{attr}_anon_low{tag}"), L(f"{pfx}_{attr}_priority_low{tag}")
            if b_ is None or A is None or P is None:
                continue
            ks = [k for k in b_ if b_[k]["correct"] and k in A and k in P]
            vals.append(np.mean([marg(A[k], tag == "_swap") - marg(P[k], tag == "_swap")
                                 for k in ks]))
        if vals:
            lows.append(float(np.mean(vals)))
            low_names.append(name)
    rng_put("ConfLow", lows)
    put("ConfLowPosN", str(sum(1 for x in lows if x > 0)))
    # "positive in all 8 cells" counted AF3, the order-sensitive cell reported separately
    # everywhere else. Every other count in the paper is stated on the primary set.
    _lp = [v for nm, v in zip(low_names, lows) if "AF3" not in nm]
    rng_put("ConfLowPrim", _lp)
    put("ConfLowPrimN", str(len(_lp)))
    put("ConfLowPrimPosN", str(sum(1 for x in _lp if x > 0)))

    # --- Sec. 5: beta against the logit of the probability each word names
    XS = [np.log(p / (1 - p)) for p in (0.1, 0.3, 0.5, 0.7, 0.9)]
    slopes, inters, r2s = [], [], []
    for name, ys in lad.items():
        pts = [(x, y) for x, y in zip(XS, ys) if y is not None]
        if len(pts) < 3:
            continue
        xa = np.array([q[0] for q in pts]); ya = np.array([q[1] for q in pts])
        A_ = np.vstack([xa, np.ones(len(xa))]).T
        (m_, b_), *_ = np.linalg.lstsq(A_, ya, rcond=None)
        slopes.append(m_); inters.append(b_)
        r2s.append(1 - np.sum((ya - A_ @ [m_, b_]) ** 2) / max(1e-12, np.sum((ya - ya.mean()) ** 2)))
    rng_put("Slope", slopes)
    rng_put("Inter", inters)
    put("RsqMin", min(r2s))

    # --- Sec. 6: the wording and clip controls
    def macro_por(pfx, attr, cond, base=None):
        b_ = L(f"{pfx}_{attr}_none") if base is None else base
        w = L(f"{pfx}_{attr}_{cond}")
        if b_ is None or w is None:
            return None
        C_ = [k for k in b_ if b_[k]["correct"] and k in w]
        per = [[w[k]["pred"] != w[k]["label"] for k in C_ if b_[k]["label"] == lb]
               for lb in ("A", "B")]
        return float(np.mean([np.mean(v) for v in per if v])) * 100

    paras, others, hedges = [], [], []
    for pfx in ("s2", "s2q3"):
        bp = L(f"{pfx}_sound_none_para")
        hi, lo = macro_por(pfx, "sound", "anon_high"), macro_por(pfx, "sound", "anon_low")
        hp, lp = macro_por(pfx, "sound", "anon_high_para", bp), macro_por(pfx, "sound", "anon_low_para", bp)
        if None not in (hi, lo, hp, lp):
            paras.append(hp - lp)
        o = macro_por(pfx, "sound", "otherclip_high")
        h = macro_por(pfx, "sound", "hedge_plain")
        if o is not None:
            others.append(o)
        if h is not None:
            hedges.append(h)
    rng_put("Para", paras, 1)
    put("OtherClipA", others[0], 1); put("OtherClipB", others[1], 1)
    put("HedgeA", hedges[0], 1); put("HedgeB", hedges[1], 1)


    # --- Sec. 6: beta outside the gate, and the largest eligible cell it is compared against
    def status(bb):
        acc = np.mean([r["correct"] for r in bb.values()])
        rec = [np.mean([r["correct"] for r in bb.values() if r["label"] == lb]) for lb in "AB"]
        if acc < 0.60: return "floor"
        if min(rec) < 0.60: return "one-sided"
        return "ceiling" if acc > 0.95 else "eligible"

    ATTRS = ["sound", "colour", "noise", "lowpass", "reverb", "silence"] + \
            [f"grad_{a}_{g}" for a in ("noise", "lowpass", "reverb") for g in ("g1","g2","g3","g4")]
    MODS = ["s2", "s2q3", "s2vl", "nm_qwen2audio", "nm_af3", "nm_nemotron", "nm_phi4mm"]
    floors, eligs = [], []
    for pfx in MODS:
        for attr in ATTRS:
            bb = L(f"{pfx}_{attr}_none")
            if bb is None: continue
            T = L(f"{pfx}_{attr}_true_high") or L(f"{pfx}_{attr}_true_high_T")
            F = L(f"{pfx}_{attr}_anon_high")
            if T is None or F is None: continue
            ks = sorted(set(bb) & set(T) & set(F))
            v = float(np.mean([0.5 * (marg(T[k]) - marg(F[k])) for k in ks]))
            st = status(bb)
            if st == "floor": floors.append(v)
            elif st == "eligible": eligs.append(v)
    if floors:
        rng_put("Floor", floors)
        put("FloorN", str(len(floors)))
        put("FloorAboveN", str(sum(1 for v in floors if v > max(eligs))))
    if eligs:
        put("EligMax", max(eligs))

    # --- Sec. 6: the answer-order audit
    def por2(pfx, attr, cond, tag):
        bb = L(f"{pfx}_{attr}_none{tag}"); w = L(f"{pfx}_{attr}_{cond}{tag}")
        if bb is None or w is None: return None
        C_ = [k for k in bb if bb[k]["correct"] and k in w]
        per = [[w[k]["pred"] != w[k]["label"] for k in C_ if bb[k]["label"] == lb] for lb in "AB"]
        return float(np.mean([np.mean(v) for v in per if v])) * 100
    for nick, pfx, attr in (("QtF", "s2", "sound"), ("QThree", "s2q3", "sound"),
                            ("QThreeN", "s2q3", "noise")):
        for tag, sfx in (("", "O"), ("_swap", "S")):
            hi, lo = por2(pfx, attr, "anon_high", tag), por2(pfx, attr, "anon_low", tag)
            if hi is not None:
                put(f"Ord{nick}Hi{sfx}", hi, 1); put(f"Ord{nick}Lo{sfx}", lo, 1)
    if "OrdQThreeNHiO" in M:
        put("OrdGapO", float(M["OrdQThreeNHiO"]) - float(M["OrdQThreeNLoO"]), 1)
        put("OrdGapS", float(M["OrdQThreeNHiS"]) - float(M["OrdQThreeNLoS"]), 1)
        # the prose said "Qwen2.5's largest gap" for two drafts while these came from
        # Qwen3/noise. Emitting the cell name as a macro means the sentence cannot drift
        # from the number again.
        put("OrdGapCell", "Qwen3/noise")
    bb = L("s2vl_sound_none_swap")
    if bb: put("VLSwapAcc", np.mean([r["correct"] for r in bb.values()]) * 100, 1)

    # --- Sec. 7: the detector, quoted once and reused
    rows = [json.loads(l) for l in open(f"{OUT}/clapx_cascade_v1.jsonl") if l.strip()]
    acc = np.mean([r["det_pred"] == r["label"] for r in rows])
    put("DetAcc", acc * 100, 1)
    put("DetAccFrac", acc, 3)
    fa = [np.mean([r["det_pred"] == r["label"] for r in rows if r["fold"] == f])
          for f in sorted({r["fold"] for r in rows})]
    put("DetFoldSD", np.std(fa) * 100, 1)
    hit, tot_ = {}, {}
    for r in rows:
        tot_[r["det_level"]] = tot_.get(r["det_level"], 0) + 1
        hit[r["det_level"]] = hit.get(r["det_level"], 0) + int(r["det_pred"] == r["label"])
    LV = ["very low", "low", "moderate", "high", "very high"]
    pr = [hit[l] / tot_[l] * 100 for l in LV if tot_.get(l)]
    put("BandMin", min(pr), 1); put("BandMax", max(pr), 1)


    # --- Sec. 7: does the loss come from the claim or from how it is packaged?
    # bare_* sends the proposition as a plain sentence; detclaim_* sends the identical
    # proposition inside "Source: ... / Claim: ...". Comparing the true-claim branch across the
    # two forms separates what the claim asserts from the packaging it arrives in.
    # order-averaged wherever both passes exist. The block branch had both from the start; the
    # bare branch's swapped pass is still running, and quoting one averaged against one not was
    # what made the block count read as 3 in one sentence and 4 in another.
    bare_gt, blk_gt, bare_pi, blk_pi = [], [], [], []
    bare_orders, bare_names = [], []
    for _n, pfx in (("Qwen2.5", "s2"), ("Qwen3", "s2q3"), ("Qwen2-Audio", "nm_qwen2audio"),
                    ("AF3", "nm_af3"), ("Nemotron", "nm_nemotron")):
        for tag, tc, fc in (("bare", "bare_true", "bare_false"),
                            ("blk", "detclaim_true", "detclaim_false")):
            gts, gfs = [], []
            for sfx in ("", "_swap"):
                bb = L(f"{pfx}_sound_none{sfx}")
                T_, F_ = L(f"{pfx}_sound_{tc}{sfx}"), L(f"{pfx}_sound_{fc}{sfx}")
                if bb is None or T_ is None or F_ is None:
                    continue
                ks = sorted(set(bb) & set(T_) & set(F_))
                a0 = np.mean([bb[k]["correct"] for k in ks])
                gts.append((np.mean([T_[k]["correct"] for k in ks]) - a0) * 100)
                gfs.append((np.mean([F_[k]["correct"] for k in ks]) - a0) * 100)
            if not gts:
                continue
            gt, gf = float(np.mean(gts)), float(np.mean(gfs))
            pi = None if gt <= 0 else (-gf / (gt - gf) if 0 < -gf / (gt - gf) < 1 else None)
            if tag == "bare":
                bare_gt.append(gt); bare_pi.append(pi); bare_orders.append(len(gts))
                bare_names.append(_n)
            else:
                blk_gt.append(gt); blk_pi.append(pi)
    put("BareOrders", str(min(bare_orders)) if bare_orders else "1")
    if bare_gt:
        rng_put("BareGain", bare_gt, 1)
        put("BareGainN", str(sum(1 for v in bare_gt if v > 0)))
        put("BareCells", str(len(bare_gt)))
        rng_put("BarePi", [v for v in bare_pi if v is not None], 3)
        put("BarePiN", str(sum(1 for v in bare_pi if v is not None)))
        # The lowest bare break-even, 0.743, is AF3's -- the order-sensitive cell the paper
        # reports separately. Quoting it as the range's low end made bare forwarding look
        # borderline-viable when on every primary cell it is not.
        _pp = [v for nm, v in zip(bare_names, bare_pi) if nm != "AF3" and v is not None]
        if _pp:
            rng_put("BarePrimPi", _pp, 3)
            put("BarePrimPiN", str(len(_pp)))
            put("BarePrimAboveN", str(sum(1 for v in _pp if v > 0.748)))
        _pg = [v for nm, v in zip(bare_names, bare_gt) if nm != "AF3"]
        if _pg:
            rng_put("BarePrimGain", _pg, 1)
            put("BarePrimGainN", str(sum(1 for v in _pg if v > 0)))
            put("BarePrimCells", str(len(_pg)))
    if blk_gt:
        rng_put("BlkGain", blk_gt, 1)
        put("BlkGainNegN", str(sum(1 for v in blk_gt if v < 0)))
        put("BlkGainPosN", str(sum(1 for v in blk_gt if v > 0)))
        put("BlkNoPiN", str(sum(1 for v in blk_pi if v is None)))
    # how many bare thresholds the detector actually clears: the abstract and Sec. 7 disagreed
    # on this once, so it is computed rather than described
    det = None
    rr = [json.loads(l) for l in open(f"{OUT}/clapx_cascade_v1.jsonl") if l.strip()]
    det = np.mean([r["det_pred"] == r["label"] for r in rr])
    above = sum(1 for v in bare_pi if v is not None and v > det)
    put("BareAboveN", str(above))
    put("BareBelowN", str(sum(1 for v in bare_pi if v is not None and v <= det)))

    # --- Sec. 4/7: the same declared confidence written as a word or as a number
    comp, comp_names = [], []
    for _n, pfx, lad in (("Qwen2.5", "s2", "cs_qwen25"), ("Qwen3", "s2q3", "cs_qwen3"),
                         ("Qwen2-Audio", "nm_qwen2audio", "nm_qwen2audio"),
                         ("AF3", "nm_af3", "nm_af3"), ("Nemotron", "nm_nemotron", "nm_nemotron")):
        bb = L(f"{pfx}_sound_none")
        G = {s: L(f"{lad}_sound_{s}") for s in ("v_vlow", "v_vlow_T", "v_high", "v_high_T",
                                                "n_015", "n_015_T", "n_085", "n_085_T")}
        if bb is None or any(v is None for v in G.values()):
            continue
        C_ = [k for k in bb if bb[k]["correct"] and all(k in v for v in G.values())]
        def swing(lf, lt, hf, ht):
            return np.mean([(0.5 * (marg(G[ht][k]) - marg(G[hf][k]))
                             - 0.5 * (marg(G[lt][k]) - marg(G[lf][k]))) for k in C_])
        w = swing("v_vlow", "v_vlow_T", "v_high", "v_high_T")
        nn = swing("n_015", "n_015_T", "n_085", "n_085_T")
        comp.append(nn / w * 100)
        comp_names.append(_n)
    if comp:
        rng_put("Compress", comp, 0)
        put("CompressN", str(len(comp)))
        _cp = [v for nm, v in zip(comp_names, comp) if nm != "AF3"]
        rng_put("CompressPrim", _cp, 0)
        put("CompressPrimN", str(len(_cp)))
        put("CompressPrimMed", float(np.median(_cp)), 0)
        put("CompressMed", float(np.median(comp)), 0)


    # --- Sec. 3: how the screened cells narrow to the analysed ones. Two reviews asked why the
    # both-order set (10) and the reported field-name set (7) differ, so the counts are derived.
    def stat_(bb):
        a_ = np.mean([r["correct"] for r in bb.values()])
        r_ = [np.mean([r["correct"] for r in bb.values() if r["label"] == lb]) for lb in "AB"]
        if a_ < 0.60: return "floor"
        if min(r_) < 0.60: return "one-sided"
        return "ceiling" if a_ > 0.95 else "eligible"
    MODS2 = ["s2","s2q3","s2vl","nm_qwen2audio","nm_af3","nm_nemotron","nm_phi4mm"]
    ATTRS2 = ["sound","colour","noise","lowpass","reverb","silence"] + \
             [f"grad_{a}_{g}" for a in ("noise","lowpass","reverb") for g in ("g1","g2","g3","g4")]
    # The five funnel counts were typed into the body for twenty-odd drafts. They were right,
    # but nothing checked them, and an unchecked count elsewhere ("five non-Qwen releases")
    # turned out to be unsupported. Derive them, and derive the both-orders denominator too:
    # reviews asked twice how 9 eligible becomes 16 run in both orders, and the answer is that
    # the 5 ceiling cells are admissible as well, plus two below-gate cells kept for the
    # position audit.
    _fun = {"floor": 0, "one-sided": 0, "ceiling": 0, "eligible": 0}
    _bothrun = 0
    for pfx in MODS2:
        for a in ATTRS2:
            o = L(f"{pfx}_{a}_none")
            if o is None: continue
            _fun[stat_(o)] += 1
            if L(f"{pfx}_{a}_none_swap") is not None: _bothrun += 1
    put("GateRunN", str(sum(_fun.values())))
    put("GateFloorN", str(_fun["floor"]))
    put("GateRecallN", str(_fun["one-sided"]))
    put("GateCeilN", str(_fun["ceiling"]))
    put("GateEligN", str(_fun["eligible"]))
    put("GateAdmissibleN", str(_fun["eligible"] + _fun["ceiling"]))
    put("GateBothRunN", str(_bothrun))
    put("GateExtraSwapN", str(_bothrun - _fun["eligible"] - _fun["ceiling"]))

    both, bothfull = 0, 0
    for pfx in MODS2:
        for a in ATTRS2:
            o, s = L(f"{pfx}_{a}_none"), L(f"{pfx}_{a}_none_swap")
            if o is None or s is None: continue
            if stat_(o) in ("eligible","ceiling") and stat_(s) in ("eligible","ceiling"):
                both += 1
                if all(L(f"{pfx}_{a}_{c}") for c in ("anon_high","anon_low","priority_high",
                       "priority_low","priority_true_high","priority_true_low")):
                    bothfull += 1
    put("BothOrderN", str(both))
    put("BothOrderFullN", str(bothfull))
    # the prose said "the other four enter the override table only"; that remainder is
    # both - bothfull and it silently went stale the moment a cell gained its battery
    put("BothOrderRestN", str(both - bothfull))

    # The one non-Qwen release screened out for answering by option position. The prose used to
    # claim five such releases and defer their names and recalls to a code release. Only Phi-4
    # was ever run -- there are no output files for the other four and no log entry recording
    # them -- and there is no code release, so the sentence now names the one model we have and
    # carries its recalls on the page, where they can be checked.
    _phi = L("nm_phi4mm_sound_none")
    if _phi is not None:
        put("PhiScreenN", str(len(_phi)))
        for lb in "AB":
            put("PhiRec" + lb,
                100 * np.mean([r["correct"] for r in _phi.values() if r["label"] == lb]), 1)

    pathlib.Path("paper/auto_all.tex").write_text(
        "% Generated by scripts/gen_all_numbers_v1.py -- do not edit by hand.\n"
        "% A number quoted in the body and absent here has been typed, which is how nine\n"
        "% figures went stale in one day. Re-run after any new condition lands.\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(M.items())))
    print(f"paper/auto_all.tex: 매크로 {len(M)}개")
    for k, v in sorted(M.items()):
        print(f"  {k:20} {v}")
