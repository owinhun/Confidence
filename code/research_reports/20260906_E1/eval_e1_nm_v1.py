#!/usr/bin/env python3
"""E1 wrapper for the nm_ runner (Qwen2-Audio etc.): adds the referent-disambiguation fields as new
conditions without touching scripts/eval_newmodels_v1.py. Output naming/resume/scoring are the runner's own.
Conditions: claimconf_{high,low}, detconf_{high,low}, readerconf_{high,low} -- false-claim block,
same key-value syntax as anon_/priority_ (Table 1 right half)."""
import sys, os
sys.path.insert(0, "/workspace/svhalluc/claude_track2/scripts")
os.chdir("/workspace/svhalluc/claude_track2")
import eval_newmodels_v1 as M
E1 = {"claimconf": "Claim confidence", "detconf": "Detector confidence", "readerconf": "Reader confidence"}
_orig = M.evidence
def evidence(cond, attr, row, claim_true=False):
    key = cond.rsplit("_", 1)[0]
    if key in E1:
        truth_yes = row["cond_label" if attr.startswith("aux_") else "label"] == "A"
        claim = M.PROP[attr](truth_yes if claim_true else not truth_yes, row)
        lvl = "high" if cond.endswith("high") else "low"
        return f'Source: anonymous system\nClaim: "{claim}"\n{E1[key]}: {lvl}\n\n'
    return _orig(cond, attr, row, claim_true)
M.evidence = evidence
M.CONDS = list(M.CONDS) + [f"{k}_{l}" for k in E1 for l in ("high", "low")]
if __name__ == "__main__":
    M.main()
