#!/usr/bin/env -S python3 -I
"""Task 5: fit_lidar_through.py on a synthetic lidar_bins.csv: open aperture bins read through 80 %, solid wall 1 %. Run from door_concept/."""
import csv, os, re, subprocess, sys, tempfile
import numpy as np
rng = np.random.default_rng(5)
d = tempfile.mkdtemp(); p = os.path.join(d, "bins.csv")
with open(p, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["cycle", "t_ms", "door", "state", "region", "obs", "thr", "dist", "side"])
    for i in range(300):
        w.writerow([i, i, "DOOR_0", "open", "ap", 40, rng.binomial(40, 0.8), 2, 1])
        w.writerow([i, i, "DOOR_0", "open", "wall", 60, rng.binomial(60, 0.01), 2, 1])
        w.writerow([i, i, "DOOR_0", "shut", "ap", 40, rng.binomial(40, 0.01), 2, 1])
tool = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "fit_lidar_through.py")
out = subprocess.run([sys.executable, "-I", tool, p, "--dry-run"], capture_output=True, text=True); print(out.stdout, out.stderr)
fails = 0
o = re.search(r"through_open=([\d.]+)", out.stdout); w_ = re.search(r"through_wall=([\d.]+)", out.stdout)
if out.returncode or not o or abs(float(o.group(1)) - 0.8) > 0.02: print("FAIL through_open"); fails += 1
if not w_ or abs(float(w_.group(1)) - 0.01) > 0.005: print("FAIL through_wall"); fails += 1
print("FAIL" if fails else "PASS"); sys.exit(1 if fails else 0)
