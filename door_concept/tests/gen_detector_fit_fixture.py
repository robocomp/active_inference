#!/usr/bin/env -S python3 -I
"""Seeded synthetic per-view log for the fit smoke test. Generating zed envelope: min .12, max .55, soft .05, hit_rate .9.
Usage: gen_detector_fit_fixture.py OUT.csv"""
import sys
import numpy as np
rng = np.random.default_rng(7)
sig = lambda x: 1 / (1 + np.exp(-x))
env = lambda f, mn, mx, s: sig((f - mn) / s) * sig((mx - f) / s)
mn, mx, s, h = 0.12, 0.55, 0.05, 0.9
out = ["cycle,door,camera,view_id,q,fill_h,fill_v,aspect,vis,kappa,hit,alive_end,crossed"]
for i in range(int(sys.argv[2]) if len(sys.argv) > 2 else 3000):
    fv = rng.uniform(0.04, 0.8); asp = rng.uniform(0.6, 1); fh = fv * asp * rng.uniform(0.6, 1.4)
    vis = rng.uniform(0.9, 1); k = rng.uniform(0.9, 1)
    q = env(fh, mn, mx, s) * env(fv, mn, mx, s) * asp * vis * k * h
    out.append(f"{i},door_{i%4+1},zed,{i},{q:.6g},{fh:.5g},{fv:.5g},{asp:.4g},{vis:.3g},{k:.3g},{int(rng.random()<q)},1,1")
open(sys.argv[1], "w").write("\n".join(out) + "\n")
