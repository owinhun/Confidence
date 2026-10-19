#!/usr/bin/env python3
"""E4 mitigation-prompt baseline: adds cascadex_warn (= cascadex_plain + a one-line caveat) to
eval_newmodels_v1 without editing it. Preregistered 2026-09-07 10:25 KST."""
import sys, os
sys.path.insert(0, "/workspace/svhalluc/claude_track2/scripts")
os.chdir("/workspace/svhalluc/claude_track2")
import eval_newmodels_v1 as M
WARN = " This tag may be wrong; judge from the recording itself."
_orig = M.evidence
def evidence(cond, attr, row, claim_true=False):
    if cond == "cascadex_warn":
        return _orig("cascadex_plain", attr, row, claim_true).rstrip("\n") + WARN + "\n\n"
    return _orig(cond, attr, row, claim_true)
M.evidence = evidence
M.CONDS = list(M.CONDS) + ["cascadex_warn"]
if __name__ == "__main__":
    M.main()
