"""Hungarian match of the final published polygon to the annotated truth, per room.

Emits one row per MATCHED corner: room, declared sigma, realised error. Surplus corners on either
side stay unmatched and are counted as structure errors — never as position error (nearest-vertex
matching instead reports a mean error of 0.43 m against a median of 0.023 m, because a spurious
corner has no truth counterpart and its 'error' is noise).
"""
import sys, glob, os, csv
import numpy as np
from scipy.optimize import linear_sum_assignment

OUT = sys.argv[1]
CORPUS = "datasets/matterport_layout/corpus594.txt"

truth = []
for line in open(CORPUS, encoding="utf-8"):
    if ";" not in line: continue
    name, pts = line.split(";", 1)
    P = [tuple(map(float, p.split(","))) for p in pts.split()]
    truth.append((name.strip(), np.array(P)))
print("truth rooms:", len(truth))

rows, per_room, failed = [], [], 0
for f in sorted(glob.glob(os.path.join(OUT, "poly_*.csv")),
                key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0])):
    r = int(f.rsplit("_", 1)[1].split(".")[0])
    if r >= len(truth): continue
    origin, last = None, None
    for line in open(f, encoding="utf-8"):
        if line.startswith("# origin"):
            origin = np.array(list(map(float, line.split()[2:4]))); continue
        if line.startswith("#"): continue
        last = line.rstrip("\n")
    if origin is None or last is None: failed += 1; continue
    p = last.split(";")
    if len(p) < 7 or not p[3].strip(): failed += 1; continue
    est = np.array([tuple(map(float, v.split(","))) for v in p[3].split()]) + origin   # map -> world
    sig = np.array([float(x) for x in p[4].split()]) if p[4].strip() else np.array([])
    closed = p[6].split(",")[0].strip() == "1"
    if not closed or len(est) == 0 or len(sig) != len(est): failed += 1; continue

    T = truth[r][1]
    D = np.linalg.norm(est[:, None, :] - T[None, :, :], axis=2)
    ri, ci = linear_sum_assignment(D)
    errs = D[ri, ci]
    for i, (a, b) in enumerate(zip(ri, ci)):
        rows.append((r, float(sig[a]), float(errs[i])))
    per_room.append((r, len(est), len(T), int(len(ri)), float(np.median(errs))))

print(f"matched corners: {len(rows)} from {len(per_room)} rooms ({failed} unusable)")
with open(os.path.join(OUT, "corners.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["room", "sigma", "error"]); w.writerows(rows)
print("wrote", os.path.join(OUT, "corners.csv"))
