#!/usr/bin/env python3
"""E4 for Nemotron: patch eval_newmodels_v1 before eval_nemotron_v1 binds its names."""
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
M.CONDS.append("cascadex_warn")
import eval_nemotron_v1 as N
if __name__ == "__main__":
    N.main()
