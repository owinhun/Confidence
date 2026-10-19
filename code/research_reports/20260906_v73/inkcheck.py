#!/usr/bin/env python3
"""Bottom ink margin per page and per column at 110 dpi (HANDOFF rule: p1-p4 107-111 px)."""
import subprocess, sys, tempfile, os, glob
from PIL import Image
import numpy as np
pdf=sys.argv[1]
with tempfile.TemporaryDirectory() as td:
    subprocess.run(["pdftoppm","-r","110","-gray",pdf,os.path.join(td,"p")],check=True)
    for f in sorted(glob.glob(os.path.join(td,"p-*.pgm"))):
        a=np.array(Image.open(f)); h,w=a.shape; ink=a<128
        rows=np.where(ink.any(axis=1))[0]
        mid=w//2
        def bottom(cols):
            r=np.where(ink[:,cols].any(axis=1))[0]; return h-1-r[-1] if r.size else None
        print(f"{os.path.basename(f)}: page bottom margin {h-1-rows[-1]:4d}px  left col {bottom(slice(0,mid))}px  right col {bottom(slice(mid,w))}px")
