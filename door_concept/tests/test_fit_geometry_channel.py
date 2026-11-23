#!/usr/bin/env -S python3 -I
"""Task 5: the geometry fit recovers mu1_sd from (geo_x, label) pairs. Synthetic frames: controls mu0, s vary; label-1 x = mu0 + 2 s + s*N(0,1),
label-0 x = mu0 + s*N(0,1). Fitted mu1_sd must be within 0.1 of 2.0, the null must read (0, 1), and too few label-1 frames keep the default.
Run from door_concept/:  python3 -I tests/test_fit_geometry_channel.py"""
import csv, os, re, subprocess, sys, tempfile
import numpy as np

rng = np.random.default_rng(11)
d = tempfile.mkdtemp()
p = os.path.join(d, "pairs.csv")
with open(p, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["cycle", "door", "label", "geo_x", "geo_mu0", "geo_s"])
    for i in range(4000):
        lab = i % 2
        mu0 = rng.uniform(0.2, 0.6); s = rng.uniform(0.03, 0.2)
        w.writerow([i, f"d{i%5}", lab, mu0 + (2.0 * s if lab else 0.0) + s * rng.standard_normal(), mu0, s])
tool = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "fit_geometry_channel.py")
out = subprocess.run([sys.executable, "-I", tool, p, "--dry-run"], capture_output=True, text=True)
print(out.stdout, out.stderr)
fails = 0
m = re.search(r"mu1_sd=([-\d.]+)", out.stdout)
if out.returncode != 0 or m is None or abs(float(m.group(1)) - 2.0) > 0.1:
    print("FAIL mu1_sd not within 0.1 of 2.0"); fails += 1
m0 = re.search(r"label 0: mean z ([-+\d.]+) sd ([\d.]+)", out.stdout)
if m0 is None or abs(float(m0.group(1))) > 0.1 or abs(float(m0.group(2)) - 1.0) > 0.1:
    print("FAIL null z not (0, 1)"); fails += 1
# too few label-1 frames: default kept, n = 0, written honestly
q = os.path.join(d, "few.csv")
with open(q, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["cycle", "door", "label", "geo_x", "geo_mu0", "geo_s"])
    for i in range(10):
        w.writerow([i, "d", 1, 1.0, 0.5, 0.1])
t = os.path.join(d, "table.csv")
open(t, "w").write("camera,param,value,n,source\nzed,hit_rate,0.9,0,x\n")
out = subprocess.run([sys.executable, "-I", tool, q, "--out", t], capture_output=True, text=True)
tab = open(t).read()
if "geometry,mu1_sd,2,0,compiled_defaults" not in tab or "zed,hit_rate,0.9,0,x" not in tab:
    print("FAIL default not kept / row not carried over:", tab); fails += 1
# no information: label-1 z ~ N(0, 1) (the channel does not separate doors): written as 0, with the real n and a "non-informative" source
r0 = os.path.join(d, "null.csv")
with open(r0, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["cycle", "door", "label", "geo_x", "geo_mu0", "geo_s"])
    for i in range(700):
        mu0 = rng.uniform(0.2, 0.6); s = rng.uniform(0.03, 0.2)
        w.writerow([i, "d", 1, mu0 + s * rng.standard_normal(), mu0, s])
t2 = os.path.join(d, "table2.csv"); open(t2, "w").write("camera,param,value,n,source\n")
subprocess.run([sys.executable, "-I", tool, r0, "--out", t2], capture_output=True, text=True)
tab = open(t2).read()
if not re.search(r"geometry,mu1_sd,0,700,.*non-informative", tab):
    print("FAIL N(0,1) samples not written as 0 / non-informative:", tab); fails += 1
print("FAIL" if fails else "PASS")
sys.exit(1 if fails else 0)
