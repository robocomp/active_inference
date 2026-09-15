#!/usr/bin/env python3
"""The §V-E corner-channel statistics grouped by PHYSICAL STOP instead of by stationary stretch.

    python3 analysis/corner_channel/by_physical_stop.py <corner_probe.csv[.gz]>

POST-HOC, AND LABELLED AS SUCH. split_and_nis.py is the pre-registered analysis and stays unmodified.
This file exists because of what its stretches turned out to be on the hardware tour (2026-09-15):

  split_and_nis.py ends a stationary stretch at the first frame whose rigid fit exceeds 0.25 deg / 5 mm.
  One such frame inside a stop splits that stop in two. The 6 hardware stretches were THREE stops:
  frames 4317-4914 broken at 4375 and 4517, and 18468-18554 broken at 18522. The viewpoint test counts
  "a corner seen in >= 3 stretches" as seen from three viewpoints, so 9 of its 10 corners were in fact
  seen from one or two places. The simulated tour has the same effect, milder: 34 stretches, 23 stops.

A stop here = stretches whose gaps are at most GAP non-still frames. The gap is SWEPT, not chosen: at
20 Hz one frame is 50 ms, too short to reach another viewpoint, while tens of frames can be a real
move. A result that changes across the sweep is reported as fragile.

Self-check: the stretch construction below is a copy of split_and_nis.py's. It is run against that
script's own printed stretch and pair counts, and the program refuses to report if they disagree.
"""
import gzip, math, re, subprocess, sys, os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_STRETCH = 25            # split_and_nis.py
GAPS = (0, 1, 5, 25, 100)   # non-still frames tolerated inside one stop; 0 = split_and_nis.py's stretches


def open_text(path):
    return gzip.open(path, "rt", encoding="utf-8", errors="ignore") if path.endswith(".gz") \
        else open(path, encoding="utf-8", errors="ignore")


def load(path):
    with open_text(path) as f:
        head = f.readline().rstrip("\n").split(",")
        idx = {c: i for i, c in enumerate(head)}
        rows = [p for p in (l.rstrip("\n").split(",") for l in f) if len(p) == len(head)]
    return idx, rows


def stretches_of(idx, rows):
    """Verbatim logic of split_and_nis.py §2: rigid fit between consecutive frames' shared detections."""
    F, M, G = idx["frame"], idx["model_index"], idx["over_gate"]
    DX, DY = idx["det_x"], idx["det_y"]
    acc = [r for r in rows if r[G] in ("0", "0.0")]
    by_det = defaultdict(dict)
    for r in acc:
        by_det[int(r[F])][int(r[M])] = (float(r[DX]), float(r[DY]))
    frames = sorted({int(r[F]) for r in rows})

    def still(a, b):
        sh = [k for k in by_det.get(a, {}) if k in by_det.get(b, {})]
        if len(sh) < 3:
            return False
        P = [by_det[a][k] for k in sh]; Q = [by_det[b][k] for k in sh]
        pcx = sum(x for x, _ in P)/len(P); pcy = sum(y for _, y in P)/len(P)
        qcx = sum(x for x, _ in Q)/len(Q); qcy = sum(y for _, y in Q)/len(Q)
        sxx = sum((x-pcx)*(u-qcx) for (x, _), (u, _) in zip(P, Q))
        sxy = sum((x-pcx)*(v-qcy) for (x, _), (_, v) in zip(P, Q))
        syx = sum((y-pcy)*(u-qcx) for (_, y), (u, _) in zip(P, Q))
        syy = sum((y-pcy)*(v-qcy) for (_, y), (_, v) in zip(P, Q))
        th = math.atan2(sxy - syx, sxx + syy)
        return abs(math.degrees(th)) < 0.25 and math.hypot(qcx-pcx, qcy-pcy) < 0.005

    s = {f for a, f in zip(frames, frames[1:]) if still(a, f)}
    out, cur = [], []
    for f in frames:
        if f in s:
            cur.append(f)
        else:
            if len(cur) >= MIN_STRETCH: out.append(cur)
            cur = []
    if len(cur) >= MIN_STRETCH: out.append(cur)
    return out, acc


def group(stretches, gap):
    stops = []
    for st in stretches:
        if stops and st[0] - stops[-1][-1][-1] - 1 <= gap:
            stops[-1].append(st)
        else:
            stops.append([st])
    return stops


def stats(stops, idx, acc):
    F, M, NX, NY = idx["frame"], idx["model_index"], idx["nu_x"], idx["nu_y"]
    by_f = defaultdict(dict)
    for r in acc:
        by_f[int(r[F])][int(r[M])] = (float(r[NX]), float(r[NY]))
    pairs = []
    for sid, stop in enumerate(stops):
        per = defaultdict(list)
        for st in stop:
            for f in st:
                for k, nu in by_f.get(f, {}).items():
                    per[k].append(nu)
        for k, v in per.items():
            if len(v) < 5: continue
            mx = sum(a for a, _ in v)/len(v); my = sum(b for _, b in v)/len(v)
            pairs.append((sid, k, math.hypot(mx, my),
                          math.sqrt(sum((a-mx)**2 + (b-my)**2 for a, b in v)/len(v)), mx, my))
    med = lambda a: sorted(a)[len(a)//2] if a else float("nan")
    bs = [p[2] for p in pairs]; ns = [p[3] for p in pairs]
    res = dict(stops=len(stops), pairs=len(pairs), repeat=med(bs), scatter=med(ns),
               dominant=100*sum(p[2] > p[3] for p in pairs)/max(len(pairs), 1))
    per_k = defaultdict(list)
    for sid, k, b, n, mx, my in pairs:
        per_k[k].append((mx, my, n))
    for m in (3, 2):
        multi = [v for v in per_k.values() if len(v) >= m]
        if not multi:
            res[f"vp{m}"] = (0, float("nan"), float("nan")); continue
        w = med([med([n for *_, n in v]) for v in multi])
        bt = []
        for v in multi:
            cx = sum(a for a, _, _ in v)/len(v); cy = sum(b for _, b, _ in v)/len(v)
            bt.append(math.sqrt(sum((a-cx)**2 + (b-cy)**2 for a, b, _ in v)/len(v)))
        res[f"vp{m}"] = (len(multi), w, med(bt))
    return res


def main(path):
    idx, rows = load(path)
    stretches, acc = stretches_of(idx, rows)

    # ── self-check against the pre-registered script ────────────────────────────────────────
    probe = path
    if path.endswith(".gz"):
        probe = os.path.join(os.environ.get("TMPDIR", "/tmp"), "by_stop_probe.csv")
        with open(probe, "w") as o, open_text(path) as i:
            o.writelines(i)
    ref = subprocess.run([sys.executable, os.path.join(HERE, "split_and_nis.py"), probe],
                         capture_output=True, text=True).stdout
    n_ref = int(re.search(r"stationary stretches .*?:\s*(\d+) of", ref).group(1))
    p_ref = int(re.search(r"(\d+) corner-stretch pairs", ref).group(1))
    mine = stats(group(stretches, 0), idx, acc)
    if (len(stretches), mine["pairs"]) != (n_ref, p_ref):
        sys.exit(f"SELF-CHECK FAILED: here {len(stretches)} stretches/{mine['pairs']} pairs, "
                 f"split_and_nis.py {n_ref}/{p_ref}. The copied logic has drifted; fix before reporting.")
    print(f"{path}\nself-check: {n_ref} stretches, {p_ref} pairs — matches split_and_nis.py\n")
    print("stretches (first..last frame, length):",
          ", ".join(f"{s[0]}..{s[-1]} ({len(s)})" for s in stretches))
    print(f"stationary frames in total: {sum(len(s) for s in stretches)}\n")

    print(f"{'gap':>4} {'stops':>5} {'pairs':>5} {'repeat':>7} {'scatter':>7} {'ratio':>6} {'dom%':>5}"
          f" | {'vp>=3: n  within  between  x':>30} | {'vp>=2: n  within  between  x':>30}")
    for g in GAPS:
        r = stats(group(stretches, g), idx, acc)
        cells = []
        for m in (3, 2):
            n, w, b = r[f"vp{m}"]
            cells.append(f"{n:8d}  {w:.4f}  {b:.4f} {b/w:5.1f}" if n else f"{n:8d}  {'-':>6}  {'-':>7} {'-':>5}")
        label = f"{g:4d}" + (" (= split_and_nis stretches)" if g == 0 else "")
        print(f"{g:4d} {r['stops']:5d} {r['pairs']:5d} {r['repeat']:7.4f} {r['scatter']:7.4f} "
              f"{r['repeat']/r['scatter']:6.1f} {r['dominant']:5.0f} | {cells[0]:>30} | {cells[1]:>30}")
    print("\ngap 0 reproduces split_and_nis.py. A viewpoint test with n < 3 corners is not a test.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
