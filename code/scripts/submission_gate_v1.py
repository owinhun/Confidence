#!/usr/bin/env python3
"""ICASSP 2027 submission gate -- every mechanical check from the 2026-09-07/08 review sessions,
run against the exact files that will be uploaded. Usage: submission_gate_v1.py 89 [date]
Exit 0 only if every hard check passes. Soft checks print WARN and do not fail the gate."""
import re, sys, os, zipfile, hashlib, subprocess, tempfile, pathlib, collections
V = sys.argv[1]; DATE = sys.argv[2] if len(sys.argv) > 2 else "2026-09-07"
REL = pathlib.Path("releases")/DATE; PDF = REL/f"icassp_v{V}.pdf"; ZIP = REL/f"icassp_v{V}.zip"
CARD = REL/f"submission_card_v{V}.txt"; KIT = pathlib.Path("template_2027_user/ex")
FAILS = []; WARNS = []
def ok(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond: FAILS.append(name)
def warn(name, detail=""): print(f"  [WARN] {name}" + (f"  ({detail})" if detail else "")); WARNS.append(name)
def run(*a): return subprocess.run(a, capture_output=True, text=True).stdout
md5 = lambda b: hashlib.md5(b).hexdigest()
def bbox(page):
    x = run("pdftotext", "-bbox", "-q", "-f", str(page), "-l", str(page), str(PDF), "-")
    return [(float(a), float(b), float(c), float(d), t) for a, b, c, d, t in
            re.findall(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</word>', x, re.S)]

print(f"== gate: {PDF}")
info = run("pdfinfo", str(PDF)); pages = int(re.search(r"Pages:\s+(\d+)", info).group(1))
print("-- PDF technical")
ok("5 pages", pages == 5, f"{pages}")
ok("US Letter 612x792", "612 x 792" in info)
ok("not encrypted", re.search(r"Encrypted:\s+no", info) is not None)
ok("<= 5 MB", PDF.stat().st_size <= 5_000_000, f"{PDF.stat().st_size} B")
fonts = [l.split() for l in run("pdffonts", str(PDF)).splitlines()[2:] if l.strip()]
ok("all fonts embedded+subset", all(f[-5] == "yes" and f[-4] == "yes" for f in fonts), f"{len(fonts)} fonts")
ok("no DejaVu (figure typeface)", not any("DejaVu" in f[0] for f in fonts))
raw = PDF.read_bytes()
ok("no raster images", len(re.findall(rb"/Subtype\s*/Image", raw)) == 0)
ok("no links/annotations", len(re.findall(rb"/Subtype\s*/Link", raw)) + len(re.findall(rb"/Annots", raw)) == 0)
if re.findall(rb"/(?:ca|CA)\s+0\.[0-9]", raw): warn("transparency present (alpha<1)", "PDF eXpress may note it; not a failure")

print("-- layout vs kit")
for p in range(1, 6):
    w = bbox(p); L = [q for q in w if (q[0]+q[2])/2 < 306]; R = [q for q in w if (q[0]+q[2])/2 >= 306]
    lb = max(q[3] for q in L)/72; rb = max(q[3] for q in R)/72
    need = 9.90 if p == 5 else (9.95 if p == 4 else 10.0)
    ok(f"p{p} columns full (L {lb:.2f} / R {rb:.2f})", (lb >= need or (p == 4 and lb >= 9.9)) and rb >= need)
    if p == 2:
        xmin = min(q[0] for q in w)/72; xmax = max(q[2] for q in w)/72
        lw = (max(q[2] for q in L) - min(q[0] for q in L))/72; gut = (min(q[0] for q in R) - max(q[2] for q in L))/72
        ok("left margin 0.756 in", abs(xmin-0.756) < 0.01, f"{xmin:.3f}")
        ok("right edge <= 7.77 in", xmax <= 7.77, f"{xmax:.3f}")
        ok("column width 3.386 in", abs(lw-3.386) < 0.01, f"{lw:.3f}")
        ok("gutter 0.236 in", abs(gut-0.236) < 0.01, f"{gut:.3f}")
        ys = sorted(set(round(q[1], 1) for q in L)); steps = collections.Counter(round(ys[i+1]-ys[i], 2) for i in range(len(ys)-1))
        lead = steps.most_common(1)[0][0]
        ok("leading <= 8 lines/inch", 72/lead <= 8.0, f"{72/lead:.2f} lines/inch")
    lo = [q[4] for q in w if q[3] > 792-60]
    ok(f"p{p} no page number", not any(re.fullmatch(r"\d{1,2}", t) for t in lo))
t1 = run("pdftotext", "-q", "-f", "1", "-l", "1", str(PDF), "-")
title = t1.strip().splitlines()[0]
ok("title ALL CAPS", title == title.upper(), title[:40])
absw = len(re.search(r"ABSTRACT\n(.*?)\nIndex Terms", t1, re.S).group(1).split())
ok("abstract <= 200 words (CMS)", absw <= 200, f"{absw}")
if absw > 150: warn("abstract > 150 words", f"{absw}; kit says 'approximately 100 to 150'")
w1 = bbox(1); h = [q for q in w1 if q[4] == "ABSTRACT"][0]; e = [q for q in w1 if q[4] == "crossing."][0]
ok("abstract block <= 3.125 in", (e[3]-h[1])/72 <= 3.125, f"{(e[3]-h[1])/72:.2f}")
t4 = run("pdftotext", "-q", "-f", "4", "-l", "4", str(PDF), "-"); t5 = run("pdftotext", "-q", "-f", "5", "-l", "5", str(PDF), "-")
ok("references do not start on p4", "REFERENCES" not in t4)
secs = [s.strip() for s in re.findall(r"^\d+\.\s+([A-Z][A-Z ]+?)\s*$", t5, re.M)]
# a vacuous pass (no headings found) must FAIL: the kit allows only these on page 5
ok("p5 = ACKNOWLEDGEMENTS + REFERENCES only", len(secs) == 2 and set(secs) == {"ACKNOWLEDGEMENTS", "REFERENCES"}, str(secs))
if "COMPLIANCE WITH ETHICAL STANDARDS" not in (t4 + t5):
    warn("no Compliance-with-Ethical-Standards section", "author decision 2026-09-08; policy says should/required but is unenforced")
full = " ".join(run("pdftotext", "-q", str(PDF), "-").split())
for tag in ("g4:", "s : sensitivity", "c : an order exceeds"):
    ok(f"table mark defined: '{tag}'", tag in full)

print("-- zip")
z = zipfile.ZipFile(ZIP); names = {n.split("/")[-1] for n in z.namelist()}
ok("zip has 7 kit files", names == {"IEEEbib.bst", "README.txt", "Template.pdf", "Template.tex", "fig1.pdf", "refs.bib", "spconf.sty"}, str(sorted(names)))
ok("spconf.sty == official kit", md5(z.read(f"icassp_v{V}/spconf.sty")) == md5((KIT/"spconf.sty").read_bytes()))
ok("IEEEbib.bst == official kit", md5(z.read(f"icassp_v{V}/IEEEbib.bst")) == md5((KIT/"IEEEbib.bst").read_bytes()))
ok("zip Template.pdf == release pdf", md5(z.read(f"icassp_v{V}/Template.pdf")) == md5(raw))
with tempfile.TemporaryDirectory() as td:
    z.extractall(td); d = pathlib.Path(td)/f"icassp_v{V}"
    for c in (["pdflatex", "-interaction=nonstopmode", "Template"], ["bibtex", "Template"], ["pdflatex", "-interaction=nonstopmode", "Template"], ["pdflatex", "-interaction=nonstopmode", "Template"]):
        subprocess.run(c, cwd=d, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log = (d/"Template.log").read_text(errors="replace")
    ok("clean-room rebuild: 5 pages", "Pages:           5" in run("pdfinfo", str(d/"Template.pdf")))
    ok("clean-room rebuild: 0 Overfull", "Overfull" not in log)
    ok("clean-room rebuild: 0 undefined", "undefined" not in log.lower())
    ok("clean-room text identical", run("pdftotext", "-q", str(d/"Template.pdf"), "-") == run("pdftotext", "-q", str(PDF), "-"))
    bbl = (d/"Template.bbl").read_text(errors="replace")
    ok("bib: no lowercase after '?'", not re.search(r"\?\s+[a-z]", bbl))

print("-- submission card")
card = CARD.read_text(encoding="utf-8")
ok("card pdf md5 matches", re.search(r"pdf md5: (\w+)", card).group(1) == md5(raw))
ok("card zip md5 matches", re.search(r"zip md5: (\w+)", card).group(1) == md5(ZIP.read_bytes()))
cabs = re.search(r"동일 문면\]\n(.*?)\n\n\[", card, re.S).group(1)
ok("card abstract has no de-hyphenation damage", "orderrobust" not in cabs and "order-robust" in cabs)
norm = lambda s: re.sub(r"[-‐-―\s]", "", s)
pabs = re.sub(r"-\n", "", re.search(r"ABSTRACT\n(.*?)\nIndex Terms", t1, re.S).group(1))
ok("card abstract == PDF abstract", norm(cabs) == norm(" ".join(pabs.split())))

print("-- generators (soft)")
lits = []
for f in list(pathlib.Path("scripts").glob("*.py")) + list(pathlib.Path("research_reports").glob("*/*.py")):
    for m in re.finditer(r"newcommand\{\{?\\\\?(\w+)\}\}?\{\{?([-+0-9.]+)\}\}?", f.read_text(errors="replace")):
        lits.append(f"{m.group(1)}={m.group(2)}")
if lits: warn("hardcoded macro literals in generators", ", ".join(lits))

print(f"\n== {len(FAILS)} FAIL, {len(WARNS)} WARN")
for f in FAILS: print("   FAIL:", f)
sys.exit(1 if FAILS else 0)
