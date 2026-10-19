#!/usr/bin/env python3
"""E1 wrapper for Nemotron: patch eval_newmodels_v1 (conditions + evidence) BEFORE eval_nemotron_v1 imports
them by name, then run eval_nemotron_v1.main(). Needs the transformers-5 env (see run_e1_nemotron.sh)."""
import sys, os
sys.path.insert(0, "/workspace/svhalluc/claude_track2/scripts"); os.chdir("/workspace/svhalluc/claude_track2")
import eval_newmodels_v1 as M
E1 = {"claimconf": "Claim confidence", "detconf": "Detector confidence", "readerconf": "Reader confidence"}
_orig = M.evidence
def evidence(cond, attr, row, claim_true=False):
    key = cond.rsplit("_", 1)[0]
    if key in E1:
        truth_yes = row["cond_label" if attr.startswith("aux_") else "label"] == "A"
        claim = M.PROP[attr](truth_yes if claim_true else not truth_yes, row); lvl = "high" if cond.endswith("high") else "low"
        return f'Source: anonymous system\nClaim: "{claim}"\n{E1[key]}: {lvl}\n\n'
    return _orig(cond, attr, row, claim_true)
M.evidence = evidence
M.CONDS.extend([f"{k}_{l}" for k in E1 for l in ("high", "low")])   # in place: eval_nemotron binds the same list
import eval_nemotron_v1 as N
if __name__ == "__main__": N.main()
