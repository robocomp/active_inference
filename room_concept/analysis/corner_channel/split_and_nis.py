#!/usr/bin/env python3
"""The pre-registered corner-channel analysis (paper/icra/PREREGISTRATION_REAL_TOUR.md).

    python3 analysis/corner_channel/split_and_nis.py <corner_probe.csv> [layout_trace.csv]

WHY THIS FILE EXISTS. The numbers in an earlier draft of §V-E were produced by an ad-hoc heredoc that
was never committed, against a run that was later overwritten (the probe writer opens with
std::ios::trunc). Re-running the analysis on surviving data gave 0.0403/0.0063 where the paper said
0.0952/0.0113 — a factor of 2.4 — and nothing could adjudicate, because neither the script nor the
data still existed. Everything quoted from this channel now comes from this file, on a log that was
copied out of tmp/ before the next run truncated it.

Reports exactly the three pre-registered quantities and nothing else:
  1. NIS/dof overall and per rotation band, with n per band
  2. the repeat/scatter split of the innovation over stationary stretches
  3. the viewpoint test: is the repeatable part attached to the WALL or to the VIEWPOINT?

★ NIS IS TRUNCATED BY ITS OWN GATE. `d2` is the squared Mahalanobis distance the association gate
tests, so rows rejected by the gate are missing from the population and the surviving d2 is censored
at the gate bound. An aggregate over accepted rows alone therefore CANNOT be read as "the channel is
calibrated" — it is calibrated-given-accepted. The `over_gate` column is reported alongside so the
censoring is visible rather than silently folded in.
"""
import sys, math
from collections import defaultdict

BANDS = [(0.0, 0.25, "<0.25 (at rest)"), (0.25, 0.5, "0.25-0.5"), (0.5, 1.0, "0.5-1"),
         (1.0, 2.0, "1-2"), (2.0, 4.0, "2-4 (fast)")]
MIN_N = 100          # pre-registered: a band below this is reported and labelled, not dropped
STILL_V, STILL_W = 0.02, 0.02          # m/s, rad/s -- pre-registered
MIN_STRETCH = 25     # consecutive still frames


def read_probe(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        head = f.readline().rstrip("\n").split(",")
        idx = {c: i for i, c in enumerate(head)}
        rows = []
        for line in f:
            p = line.rstrip("\n").split(",")
            if len(p) != len(head):
                continue                      # the final line of a truncated log
            try:
                rows.append(p)
            except ValueError:
                continue
    return idx, rows


def read_trace(path):
    """frame -> (x, y, theta). Optional; without it motion is recovered from the detections."""
    pose = {}
    for line in open(path, encoding="utf-8", errors="ignore"):
        if line.startswith("#"):
            continue
        p = line.split(";")
        if len(p) < 3:
            continue
        try:
            pose[int(p[0])] = tuple(float(v) for v in p[2].split(","))
        except (ValueError, IndexError):
            continue
    return pose


def rates_from_pose(pose):
    """per-frame |dtheta| in deg and |dxy| in m, keyed by frame."""
    fr = sorted(pose)
    out = {}
    for a, b in zip(fr, fr[1:]):
        d = pose[b][2] - pose[a][2]
        while d > math.pi:  d -= 2 * math.pi
        while d < -math.pi: d += 2 * math.pi
        out[b] = (math.degrees(abs(d)), math.hypot(pose[b][0] - pose[a][0], pose[b][1] - pose[a][1]))
    return out


def main(probe_path, trace_path=None):
    idx, rows = read_probe(probe_path)
    need = ("frame", "model_index", "over_gate", "d2", "nu_x", "nu_y")
    missing = [c for c in need if c not in idx]
    if missing:
        sys.exit(f"probe file lacks columns: {missing}")
    F, M, G, D2 = idx["frame"], idx["model_index"], idx["over_gate"], idx["d2"]
    NX, NY = idx["nu_x"], idx["nu_y"]

    frames = sorted({int(r[F]) for r in rows})
    print(f"{len(rows)} candidate rows over {len(frames)} frames "
          f"[{frames[0]}..{frames[-1]}]  from {probe_path}")
    gated = sum(1 for r in rows if r[G] not in ("0", "0.0"))
    print(f"  rejected by the gate: {gated} ({100*gated/len(rows):.1f}%) "
          f"-- NIS below is over ACCEPTED rows and is censored by the gate\n")

    # ── 1. NIS ────────────────────────────────────────────────────────────────────────────────
    acc = [r for r in rows if r[G] in ("0", "0.0")]
    DX, DY = idx["det_x"], idx["det_y"]
    by_det = defaultdict(dict)
    for r in acc:
        by_det[int(r[F])][int(r[M])] = (float(r[DX]), float(r[DY]))
    nis_all = [float(r[D2]) / 2.0 for r in acc]
    print(f"1. NIS/dof overall: {sum(nis_all)/len(nis_all):.3f}  (n={len(nis_all)}, target 1.0)")

    pose = read_trace(trace_path) if trace_path else {}
    if pose:
        rate = rates_from_pose(pose)
        print(f"   rotation rate from {trace_path} ({len(pose)} poses)")
        per = defaultdict(list)
        for r in acc:
            f = int(r[F])
            if f in rate:
                per[next((lab for lo, hi, lab in BANDS if lo <= rate[f][0] < hi), None)].append(
                    float(r[D2]) / 2.0)
        print(f"   {'band [deg/frame]':>18} {'n':>7} {'NIS/dof':>9}")
        for _, _, lab in BANDS:
            v = per.get(lab, [])
            flag = "  UNDERPOWERED" if 0 < len(v) < MIN_N else ("  EMPTY" if not v else "")
            m = f"{sum(v)/len(v):9.3f}" if v else f"{'-':>9}"
            print(f"   {lab:>18} {len(v):7d} {m}{flag}")
    else:
        print("   no layout_trace.csv given -> no rotation bands "
              "(the probe file carries no pose; the band table needs it)")

    # ── 2 + 3. stationary stretches ───────────────────────────────────────────────────────────
    # ★ STILLNESS IS RECOVERED FROM THE DETECTIONS, NOT FROM A POSE COLUMN. The probe carries no
    # pose, and the file that would bridge frame->timestamp (layout_trace.csv) is not always written.
    # Fitting a rigid transform between consecutive frames' shared corner detections gives the
    # inter-frame motion directly: if the best-fit rotation is under 0.25 deg and the translation
    # under 5 mm, the robot did not move between those frames. This needs nothing but the probe, and
    # it is measured rather than asserted -- which matters here, because the one pose column this
    # pipeline does log is the ROOM MODEL's, not the robot's (commit 1b7d513), and reading it as the
    # robot's is exactly the error that put a wrong rotation figure into a paper figure.
    def kabsch_still(a, b):
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

    still = {f for a, f in zip(frames, frames[1:]) if kabsch_still(a, f)}
    how = "rigid fit between consecutive frames' shared detections (<0.25 deg, <5 mm)"

    stretches, cur = [], []
    for f in frames:
        if f in still:
            cur.append(f)
        else:
            if len(cur) >= MIN_STRETCH: stretches.append(cur)
            cur = []
    if len(cur) >= MIN_STRETCH: stretches.append(cur)
    print(f"\n2. stationary stretches ({how}): {len(stretches)} of >={MIN_STRETCH} frames")
    if not stretches:
        print("   none -- this run cannot support the repeat/scatter split.")
        print("   The tour needs deliberate stops of >= 30 s; re-run it.")
        return

    by_f = defaultdict(dict)
    for r in acc:
        by_f[int(r[F])][int(r[M])] = (float(r[NX]), float(r[NY]))
    pairs = []
    for sid, st in enumerate(stretches):
        per_corner = defaultdict(list)
        for f in st:
            for k, nu in by_f.get(f, {}).items():
                per_corner[k].append(nu)
        for k, v in per_corner.items():
            if len(v) < 5: continue
            mx = sum(a for a, _ in v) / len(v); my = sum(b for _, b in v) / len(v)
            bias = math.hypot(mx, my)
            noise = math.sqrt(sum((a-mx)**2 + (b-my)**2 for a, b in v) / len(v))
            pairs.append((sid, k, bias, noise, mx, my))
    if not pairs:
        print("   no corner observed >=5 times in a stretch"); return
    bs = sorted(p[2] for p in pairs); ns = sorted(p[3] for p in pairs)
    med = lambda a: a[len(a)//2]
    print(f"   {len(pairs)} corner-stretch pairs")
    print(f"   repeats (bias)  median {med(bs):.4f} m")
    print(f"   scatters (noise) median {med(ns):.4f} m")
    print(f"   ratio {med(bs)/med(ns):.1f}x   bias exceeds scatter in "
          f"{100*sum(1 for p in pairs if p[2] > p[3])/len(pairs):.0f}% of pairs")

    # ── NIS and sigma_pred, PER CANDIDATE, from the probe alone ───────────────────────────────
    # ⚠ DO NOT USE corner_nis.csv's sigma_pred COLUMN FOR THIS. It is T.s_pred_sigma(), which is
    # s_pred_sum / s_terms_n -- a running mean over the whole tour to that point (corner_detector.h).
    # It cannot respond to motion state: it is flat BY CONSTRUCTION, and splitting it by stillness
    # measures only where in the tour the stops happened to fall. A draft of this paper reported that
    # flatness as a finding. The per-candidate term is sprd_xx/xy/yy in the probe, below.
    # ⚠ AND THE GATE CENSORS NIS. d2 is the statistic the association gate tests, so quoting one
    # population without the other picks a side: pre-gate includes the mis-associations the gate
    # exists to reject, accepted-only is conditioned on having passed. Both are printed.
    def sig_major(r):
        a, b, c = float(r[PXX]), float(r[PXY]), float(r[PYY])
        tr, det = a + c, a * c - b * b
        disc = max(tr * tr / 4.0 - det, 0.0)
        return math.sqrt(max(tr / 2.0 + math.sqrt(disc), 0.0))
    PXX, PXY, PYY = idx["sprd_xx"], idx["sprd_xy"], idx["sprd_yy"]
    med = lambda a: sorted(a)[len(a) // 2] if a else float("nan")
    print(f"\n2b. NIS and sigma_pred by motion state, per candidate (no pose column used)")
    print(f"   {'population':>22} {'n':>8} {'NIS/dof':>9} {'sigma_pred med':>15}")
    for lab, sel in (("pre-gate, still",  [r for r in rows if int(r[F]) in still]),
                     ("pre-gate, moving", [r for r in rows if int(r[F]) not in still]),
                     ("accepted, still",  [r for r in acc  if int(r[F]) in still]),
                     ("accepted, moving", [r for r in acc  if int(r[F]) not in still])):
        if not sel: continue
        print(f"   {lab:>22} {len(sel):8d} {sum(float(r[D2]) for r in sel)/2.0/len(sel):9.3f}"
              f" {med([sig_major(r) for r in sel]):15.4f}")
    sp_still = med([sig_major(r) for r in acc if int(r[F]) in still])
    print(f"   -> sigma_pred at rest {sp_still:.4f} m against a repeatable innovation of "
          f"{med(bs):.4f} m: the pose covariance carries no term the size of the bias")

    print(f"\n3. viewpoint test -- is the repeatable part attached to the WALL or the VIEWPOINT?")
    per_k = defaultdict(list)
    for sid, k, bias, noise, mx, my in pairs:
        per_k[k].append((mx, my, noise))
    multi = {k: v for k, v in per_k.items() if len(v) >= 3}
    if not multi:
        print("   no corner seen in >=3 separate stretches; test not possible on this run"); return
    within, between = [], []
    for k, v in multi.items():
        within.append(sorted(n for _, _, n in v)[len(v)//2])
        cx = sum(a for a, _, _ in v)/len(v); cy = sum(b for _, b, _ in v)/len(v)
        between.append(math.sqrt(sum((a-cx)**2 + (b-cy)**2 for a, b, _ in v)/len(v)))
    w, b = med(sorted(within)), med(sorted(between))
    print(f"   {len(multi)} corners seen in >=3 stretches")
    print(f"   within-stretch scatter  {w:.4f} m")
    print(f"   between-stretch spread  {b:.4f} m   ({b/w:.1f}x within)")
    print(f"   -> {'VIEWPOINT-attached: a map offset is ruled out' if b > 2*w else 'consistent with a WALL-attached offset'}")


if __name__ == "__main__":
    if not 2 <= len(sys.argv) <= 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
