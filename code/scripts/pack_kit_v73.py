#!/usr/bin/env python3
"""Flatten wrapper + generated macros + body into a kit-style single Template.tex, build it in a
clean temp dir, verify the text matches the working build, and write releases/<date>/icassp_vN.{zip,pdf}
plus a regenerated submission card. Usage: pack_kit_v73.py 73 [date]"""
import re, sys, os, shutil, subprocess, tempfile, zipfile, hashlib, datetime, pathlib
V=sys.argv[1]; DATE=sys.argv[2] if len(sys.argv)>2 else datetime.date.today().isoformat()
P=pathlib.Path("paper"); body=(P/f"icassp_body_confidence_v{V}.tex").read_text(); wrap=(P/f"icassp_confidence_v{V}.tex").read_text()
autos=re.findall(r"\\input\{(auto_[a-z0-9]+)\}", body)
gen="% ====== Generated numbers -- do not edit by hand (see repo scripts/) ======\n"
for a in autos: gen+=f"% ---- from {a}.tex ----\n"+(P/f"{a}.tex").read_text().rstrip()+"\n"
flat=re.sub(r"\\input\{auto_[a-z0-9]+\}\n","",body)
bib=re.search(r"\\bibliography\{(\w+)\}",flat).group(1)
flat=flat.replace(f"\\bibliography{{{bib}}}","\\bibliography{refs}")
fig=re.search(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}",flat).group(1)
flat=flat.replace(f"{{{fig}}}","{fig1.pdf}")
tmpl=wrap.replace(f"\\input{{icassp_body_confidence_v{V}.tex}}", flat)
tmpl=tmpl.replace("\\begin{document}", gen+"\\begin{document}",1)
nref=len(re.findall(r"\\bibitem",(P/f"icassp_confidence_v{V}.bbl").read_text()))
with tempfile.TemporaryDirectory() as td:
    k=pathlib.Path(td)/f"icassp_v{V}"; k.mkdir()
    (k/"Template.tex").write_text(tmpl)
    for src,dst in [(P/"spconf.sty","spconf.sty"),(P/"IEEEbib.bst","IEEEbib.bst"),(P/f"{bib}.bib","refs.bib"),(P/fig,"fig1.pdf")]:
        shutil.copy(src,k/dst)
    for cmd in (["pdflatex","-interaction=nonstopmode","Template"],["bibtex","Template"],["pdflatex","-interaction=nonstopmode","Template"],["pdflatex","-interaction=nonstopmode","Template"]):
        subprocess.run(cmd,cwd=k,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    t1=subprocess.run(["pdftotext","-layout",str(k/"Template.pdf"),"-"],capture_output=True,text=True).stdout
    t2=subprocess.run(["pdftotext","-layout",str(P/f"icassp_confidence_v{V}.pdf"),"-"],capture_output=True,text=True).stdout
    assert t1==t2, "clean-room build text differs from working build"
    pages=re.search(r"Pages:\s+(\d+)",subprocess.run(["pdfinfo",str(k/"Template.pdf")],capture_output=True,text=True).stdout).group(1)
    md5=lambda p: hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest()
    _ti = re.search(r"\\title\{(.*?)\}\n", tmpl, re.S)
    _title = re.sub(r"\s+", " ", _ti.group(1)).strip() if _ti else "(title not found)"
    (k/"README.txt").write_text(f"""ICASSP 2027 submission -- "{_title}"
Paper version v{V} ({DATE}).  Laid out like the official ICASSP 2027 Paper Kit:

  Template.tex   the whole paper, single file (kit-style main)
  refs.bib       {nref} references
  fig1.pdf       Figure 1
  spconf.sty     official kit style  (md5 {md5(k/'spconf.sty')})
  IEEEbib.bst    official kit bib style (md5 {md5(k/'IEEEbib.bst')})
  Template.pdf   the built paper -- {pages} pages (4 technical + ack/references)

BUILD   pdflatex Template ; bibtex Template ; pdflatex Template ; pdflatex Template
Built from an empty directory; the PDF text is identical to paper/icassp_confidence_v{V}.pdf in the
working repo, where the version history (outputs/log_v1.md) and generator scripts live.

STATE   Authors, affiliations and funding are in the PDF; page 5 carries the funding
acknowledgement and the references (both allowed there by the live 2027 paper kit). By author
decision (2026-09-08) there is no Compliance-with-Ethical-Standards section: the SPS policy page
says "should include" plus one "required" sentence, no consequence is stated, the editorial
policies do not list it, the kit says the 5th page "may" carry it, and accepted ICASSP 2025/2026
papers omit it.  Deadline {'2026-09-16'}; upload Template.pdf (single-anonymous review).

The \\newcommand block before \\begin{{document}} holds the generated numbers: every figure quoted in
the paper is defined there and nowhere else. Rebuilt by scripts/*.py in the repo; do not hand-edit.
""")
    for junk in k.glob("Template.*"):
        if junk.suffix not in (".tex",".pdf"): junk.unlink()
    rel=pathlib.Path("releases")/DATE; rel.mkdir(parents=True,exist_ok=True)
    zp=rel/f"icassp_v{V}.zip"
    with zipfile.ZipFile(zp,"w",zipfile.ZIP_DEFLATED) as z:
        for f in sorted(k.iterdir()): z.write(f,f"icassp_v{V}/{f.name}")
    shutil.copy(k/"Template.pdf",rel/f"icassp_v{V}.pdf")
# submission card: abstract and title taken from the built PDF's text, not from an older card
txt=subprocess.run(["pdftotext","-f","1","-l","1",str(pathlib.Path("releases")/DATE/f"icassp_v{V}.pdf"),"-"],capture_output=True,text=True).stdout
# The abstract for the CMS form comes from the SOURCE, not from pdftotext. Reading it back out of
# the PDF meant undoing LaTeX's end-of-line hyphenation with re.sub("-\n",""), which cannot tell
# "de-\nclares" (join) from "order-\nrobust" (keep the hyphen) -- v88's card shipped "orderrobust"
# into the field that is pasted verbatim into the submission form.
_asrc=re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}",body,re.S).group(1)
_asrc=re.sub(r"(?<!\\)%.*","",_asrc)
for _a,_b in (("---","\u2014"),("--","\u2013"),("``","\u201c"),("''","\u201d"),
              ("'","\u2019"),("\\&","&"),("~"," ")):
    _asrc=_asrc.replace(_a,_b)
abs_=re.sub(r"\s+"," ",_asrc).strip()
assert "\\" not in abs_, f"unexpanded LaTeX in the card abstract: {abs_}"
# Cross-check against what the PDF actually prints, ignoring hyphens so the check survives both
# LaTeX hyphenation and real compounds; any other difference is a real drift and must fail here.
_apdf=re.search(r"ABSTRACT\n(.*?)\nIndex Terms",txt,re.S).group(1)
_apdf=re.sub(r"\s+"," ",re.sub(r"-\n","",_apdf)).strip()
_norm=lambda t: re.sub(r"[-\u2010-\u2015\s]","",t)
assert _norm(_apdf)==_norm(abs_), "card abstract differs from the abstract the PDF prints"
kw=re.search(r"Index Terms—\s*(.*?)\n\s*1\. INTRODUCTION",txt,re.S).group(1); kw=re.sub(r"\s+"," ",kw).strip()
title=re.search(r"\\title\{(.*?)\}",body,re.S).group(1); title=re.sub(r"\s+"," ",title).strip()
# Authors are derived from the wrapper, never hand-typed: the kit FAQ rejects a submission whose
# PDF author list differs from the online form, and this card carried a stale two-author list into
# v76 (which has three) the one time it was hardcoded.
def _braced(src, macro):
    """Body of \\macro{...} with balanced braces: \\name now nests a \\thanks{...}."""
    i = src.index("\\" + macro + "{") + len(macro) + 2
    d, j = 1, i
    while d:
        if src[j] == "{": d += 1
        elif src[j] == "}": d -= 1
        j += 1
    return src[i:j - 1]
_nm = _braced(wrap, "name")
_ad = _braced(wrap, "address")
# superscripts may carry an equal-contribution or corresponding-author mark: $^{1,*}$, $^{2,\\dagger}$
_names = re.findall(r"([A-Z][A-Za-z.\- ]+?)\$\^\{(\d+)[^}]*\}\$", _nm)
_marks = dict((n.strip(), m) for n, m in
              re.findall(r"([A-Z][A-Za-z.\- ]+?)\$\^\{\d+,\s*([^}]+)\}\$", _nm))
_affil = dict((k, re.sub(r"\s+", " ", v).strip(" \\\n"))
              for k, v in re.findall(r"\$\^\{(\d+)\}\$([^$]+)", _ad))
assert _names, "no authors parsed from the wrapper"
_sym = {"*": " [equal contribution]", "\\dagger": " [corresponding author]"}
authors = "\n".join(
    f"{i}. {n.strip():<14s} -- {_affil[k]}{_sym.get(_marks.get(n.strip(), ''), '')}"
    for i, (n, k) in enumerate(_names, 1))
zm,pm=hashlib.md5(zp.read_bytes()).hexdigest(),hashlib.md5((rel/f"icassp_v{V}.pdf").read_bytes()).hexdigest()
(rel/f"submission_card_v{V}.txt").write_text(f"""ICASSP 2027 제출 카드 — v{V} ({DATE})
=====================================================
[업로드 파일] releases/{DATE}/icassp_v{V}.pdf  (zip 안 Template.pdf와 동일)
  zip md5: {zm}
  pdf md5: {pm}

[제목 -- 폼에 정확히]
{title}

[저자 -- 순서·표기 PDF와 글자 일치 필수. kit FAQ: 불일치는 반려 사유]
{authors}

[CMS contact author] Unsang Park -- 폼에서도 지정할 것 (PDF의 dagger 각주와 별개 절차)

[초록 -- {len(abs_.split())}단어, v{V} PDF와 동일 문면]
{abs_}

[Index Terms]
{kw}

[체크리스트]
- [ ] 논문 제출: 2026-09-16
- [ ] ORCiD validation (저자 전원 본인 직접): 2026-09-23  ** 미이행 시 자동 철회 **
- [ ] (억셉 후) 저자 1인 등록+결제: 2027-02-10
- [ ] 윤리/COI 문구 없음 -- 저자 결정(정책 should+required 1문장, 미시행; 억셉 논문들도 없음)\n- [ ] 사사(IITP 과제명 2건 포함) 5쪽 §11 -- 저자 표준 문안 그대로
""")
print(f"OK releases/{DATE}/icassp_v{V}.zip md5 {zm}\n   releases/{DATE}/icassp_v{V}.pdf md5 {pm}  pages {pages}  refs {nref}")
