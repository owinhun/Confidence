#!/usr/bin/env python3
"""E1 wrapper for the Qwen3-Omni stage-2 runner (s2q3_ outputs); adds claimconf_/detconf_/readerconf_ conditions."""
import sys, os
sys.path.insert(0, "/workspace/svhalluc/claude_track2/scripts"); os.chdir("/workspace/svhalluc/claude_track2")
import eval_stage2_q3_v1 as M
E1 = {"claimconf": "Claim confidence", "detconf": "Detector confidence", "readerconf": "Reader confidence"}
_orig = M.evidence
def evidence(cond, attr, row):
    key = cond.rsplit("_", 1)[0]
    if key in E1:
        truth_yes = row["cond_label" if attr.startswith("aux_") else "label"] == "A"
        claim = M.PROP[attr](not truth_yes, row); lvl = "high" if cond.endswith("high") else "low"
        return f'Source: anonymous system\nClaim: "{claim}"\n{E1[key]}: {lvl}\n\n'
    return _orig(cond, attr, row)
M.evidence = evidence
M.CONDS = list(M.CONDS) + [f"{k}_{l}" for k in E1 for l in ("high", "low")]
if __name__ == "__main__": M.main()
