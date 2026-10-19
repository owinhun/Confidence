#!/usr/bin/env python3
"""Every number in the manuscript, listed with its source.

Piecemeal checking is what let six stale figures through: each pass looked at whatever the last
review mentioned. This enumerates instead. Numbers reaching the PDF come from three places --
generator macros, the table macros, and hard-typed prose -- and only the third can silently rot.
The job here is to make that third list explicit and short.
"""

from __future__ import annotations

import pathlib
import re
import sys

PAPER = pathlib.Path("paper")
import sys as _sys
# the path was pinned to v9, so every later run silently checked a stale manuscript and
# reported its hardcoded numbers as the current ones
BODY = (pathlib.Path(_sys.argv[1]) if len(_sys.argv) > 1
        else max(PAPER.glob("icassp_body_confidence_v*.tex"),
                 key=lambda q: int(q.stem.rsplit("_v", 1)[1])))
# a fixed list goes stale every time an analysis adds a macro file; take whatever the
# manuscript actually \input{}s instead
AUTO = None  # resolved from the body below


def macros_defined():
    d = {}
    for f in auto:
        p = PAPER / f
        if not p.exists():
            continue
        for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{(.*?)\}\s*$", p.read_text(), re.M):
            d[m.group(1)] = m.group(2)
    return d


def strip_floats(t):
    t = re.sub(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", "<TABLE>", t, flags=re.S)
    t = re.sub(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", "<FIGURE>", t, flags=re.S)
    return "\n".join(l for l in t.splitlines() if not l.strip().startswith("%"))


if __name__ == "__main__":
    body = BODY.read_text()
    auto = [f"{n}.tex" for n in re.findall(r"\\input\{(auto_[a-z0-9]+)\}", body)]
    mac = macros_defined()
    prose = strip_floats(body)

    used = set(re.findall(r"\\([A-Z]\w+)\b", prose))
    used_macros = sorted(m for m in used if m in mac)
    print(f"정의된 생성기 매크로 {len(mac)}개, 산문에서 쓰는 것 {len(used_macros)}개\n")

    # any macro referenced but not defined would render as an undefined control sequence
    missing = sorted(m for m in used if m not in mac and m[0].isupper()
                     and m not in {"TABLE", "FIGURE"})
    if missing:
        print("!! 정의되지 않은 매크로 후보:", missing[:10], "\n")

    # hard-typed decimals left in prose
    hard = []
    for i, line in enumerate(prose.splitlines(), 1):
        for m in re.finditer(r"(?<![\\\w])(\d+\.\d+)", line):
            hard.append((i, m.group(1), line.strip()))
    print(f"산문에 남은 하드코딩 소수 {len(hard)}개:\n")
    seen = set()
    for i, v, line in hard:
        key = (v, line[:40])
        if key in seen:
            continue
        seen.add(key)
        print(f"  L{i:<4} {v:>8}   {line[:110]}")
