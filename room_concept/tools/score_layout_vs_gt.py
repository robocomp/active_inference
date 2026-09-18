#!/usr/bin/env python3
"""
score_layout_vs_gt.py — grade an ESTIMATE-mode (wall-SLAM) run against a known rectangular room.

The estimated room frame is re-anchored by the agent (bbox centre, theta0 = 0), so it is NOT the Webots
world frame. Everything that compares geometry is therefore RIGID-INVARIANT: side lengths, turn angles,
area, and an IoU / corner error taken after the best SE(2) alignment to the ground-truth rectangle. The
pose channel is graded the same way: est = R·gt + t is fitted per re-anchor segment, and what is scored is
the residual of that fit, never a raw difference between two frames that were never meant to agree.

Ground truth defaults to tworooms-piso.wbt Room2: interior 6.00 x 4.00 m (inner wall faces).

  python3 tools/score_layout_vs_gt.py                       # tmp/layout_trace.csv + gt_error.csv
  python3 tools/score_layout_vs_gt.py --gt 6.0 4.0 --since-ms 1789...   # only rows after a timestamp

Reads tmp/layout_trace.csv (';' separated: frame;ts_ms;room_x,room_y,room_phi;verts;corner_sigma;
edge_sigma_d;closed,publishable) and tmp/sdf_localizer/gt_error.csv.
"""
import argparse
import csv
import math
import os
import sys

import numpy as np
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon

KINK_DEG = 30.0   # a vertex turning less than this is a split wall, not a corner (reporting only)


def load_layout(path, since_ms):
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            p = line.rstrip('\n').split(';')
            if len(p) < 7:
                continue
            try:
                ts = int(p[1])
                if since_ms is not None and ts < since_ms:
                    continue
                verts = [tuple(map(float, v.split(','))) for v in p[3].split()]
                csig = [float(x) for x in p[4].split()] if p[4].strip() else []
                closed, publishable = (int(x) for x in p[6].split(','))
            except (ValueError, IndexError):
                continue
            rows.append(dict(frame=int(p[0]), ts=ts, verts=verts, csig=csig,
                             closed=closed, publishable=publishable))
    return rows


def turns_deg(verts):
    n = len(verts)
    out = []
    for i in range(n):
        p0, p1, p2 = verts[i - 1], verts[i], verts[(i + 1) % n]
        a = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        b = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        out.append(math.degrees((b - a + math.pi) % (2 * math.pi) - math.pi))
    return out


def sides(verts):
    n = len(verts)
    return [math.dist(verts[i], verts[(i + 1) % n]) for i in range(n)]


def best_alignment(poly, gt):
    """Max IoU over rotation (coarse 1 deg then 0.1 deg) with centroids aligned. Returns (iou, deg, aligned)."""
    c = poly.centroid
    base = translate(poly, -c.x, -c.y)
    best = (-1.0, 0.0, base)
    for deg in np.arange(0.0, 360.0, 1.0):
        q = rotate(base, deg, origin=(0, 0))
        iou = q.intersection(gt).area / q.union(gt).area
        if iou > best[0]:
            best = (iou, deg, q)
    lo = best[1]
    for deg in np.arange(lo - 1.0, lo + 1.0, 0.1):
        q = rotate(base, deg, origin=(0, 0))
        iou = q.intersection(gt).area / q.union(gt).area
        if iou > best[0]:
            best = (iou, deg, q)
    return best


def corner_errors(aligned, gt, csig):
    """Nearest-GT-corner error for each estimated vertex, plus the vertex's own published sigma."""
    gtc = list(gt.exterior.coords)[:-1]
    est = list(aligned.exterior.coords)[:-1]
    out = []
    for i, v in enumerate(est):
        e = min(math.dist(v, g) for g in gtc)
        s = csig[i] if i < len(csig) else float('nan')
        out.append((e, s))
    return out


def grade_layout(row, gt):
    v = row['verts']
    if len(v) < 3:
        return None
    poly = Polygon(v)
    if not poly.is_valid:
        poly = poly.buffer(0)
    iou, deg, aligned = best_alignment(poly, gt)
    tr = turns_deg(v)
    return dict(n=len(v), turns=tr, kinks=sum(1 for t in tr if abs(t) < KINK_DEG),
                sides=sides(v), area=poly.area, iou=iou, rot=deg,
                corners=corner_errors(aligned, gt, row['csig']))


def walls_timeline(path, since_ms):
    """Per frame, the latest (phi, d, sigma_d, frames_seen) of every wall. Hesse form n.p = d, n=(cos phi, sin phi)."""
    by_ts = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                t = int(r['ts_ms'])
                if since_ms is not None and t < since_ms:
                    continue
                by_ts.setdefault(t, {})[r['wall_id']] = (float(r['phi_rad']), float(r['d_m']),
                                                         float(r['sigma_d']), int(r['frames_seen']))
            except (ValueError, KeyError):
                continue
    return sorted(by_ts.items())


def polygon_from_walls(walls):
    """Convex room from its walls: order by normal angle, intersect neighbours. Exactly what a 4-wall
    rectangle needs; returns None for anything else (a non-convex room cannot be ordered this way)."""
    if len(walls) < 3:
        return None
    ws = sorted(walls.values(), key=lambda w: w[0] % (2 * math.pi))
    verts = []
    for i in range(len(ws)):
        (p1, d1, *_), (p2, d2, *_) = ws[i], ws[(i + 1) % len(ws)]
        a = np.array([[math.cos(p1), math.sin(p1)], [math.cos(p2), math.sin(p2)]])
        if abs(np.linalg.det(a)) < 1e-3:
            return None
        verts.append(tuple(np.linalg.solve(a, np.array([d1, d2]))))
    return verts


def fit_se2(src, dst):
    """Least-squares R, t with dst ~ R·src + t (2-D Umeyama, no scale)."""
    ms, md = src.mean(0), dst.mean(0)
    H = (src - ms).T @ (dst - md)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return R, md - R @ ms


def grade_pose(path, since_ms):
    if not os.path.exists(path):
        return None
    ts, g, e, gth, eth = [], [], [], [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                t = int(r['ts_ms'])
                if since_ms is not None and t < since_ms:
                    continue
                ts.append(t)
                g.append((float(r['gt_x']), float(r['gt_y'])))
                e.append((float(r['est_x']), float(r['est_y'])))
                gth.append(float(r['gt_theta'])); eth.append(float(r['est_theta']))
            except (ValueError, KeyError):
                continue
    if len(ts) < 20:
        return None
    g, e = np.array(g), np.array(e)
    # Segment at frame changes: an estimated-position jump the ground truth does not share.
    dg = np.linalg.norm(np.diff(g, axis=0), axis=1)
    de = np.linalg.norm(np.diff(e, axis=0), axis=1)
    cuts = [0] + [i + 1 for i in range(len(de)) if de[i] > 0.5 and de[i] > 5 * dg[i] + 0.2] + [len(ts)]
    segs = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a < 20:
            continue
        R, t = fit_se2(g[a:b], e[a:b])
        res = np.linalg.norm((g[a:b] @ R.T + t) - e[a:b], axis=1)
        off = np.unwrap(np.array(gth[a:b])) - np.unwrap(np.array(eth[a:b]))
        # WINDOWED fits as well. A re-anchor below the jump detector, or a slow phantom crawl, is NOT a jump,
        # and one SE(2) fit across it reports the whole segment as bad. Per-window fits separate "one bad
        # window" (a missed frame change) from "every window bad" (the track itself is wrong).
        win = []
        wa = a
        while wa < b:
            wb = wa
            while wb < b and ts[wb] - ts[wa] < 20000:
                wb += 1
            if wb - wa >= 20:
                Rw, tw = fit_se2(g[wa:wb], e[wa:wb])
                rw = np.linalg.norm((g[wa:wb] @ Rw.T + tw) - e[wa:wb], axis=1)
                win.append(float(np.sqrt((rw ** 2).mean())))
            wa = wb
        segs.append(dict(t0=ts[a], t1=ts[b - 1], n=b - a, rms=float(np.sqrt((res ** 2).mean())),
                         p95=float(np.percentile(res, 95)), yaw_sd_deg=float(np.degrees(off.std())),
                         travel=float(np.linalg.norm(np.diff(g[a:b], axis=0), axis=1).sum()),
                         win_med=float(np.median(win)) if win else float('nan'),
                         win_max=float(max(win)) if win else float('nan'), win_n=len(win)))
    return dict(n=len(ts), jumps=len(cuts) - 2, segs=segs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--layout', default='tmp/layout_trace.csv')
    ap.add_argument('--gt-error', default='tmp/sdf_localizer/gt_error.csv')
    ap.add_argument('--gt', nargs=2, type=float, default=[6.00, 4.00], metavar=('LX', 'LY'),
                    help='ground-truth rectangle interior, metres (default: Room2 of tworooms-piso)')
    ap.add_argument('--since-ms', type=int, default=None, help='ignore rows before this ts_ms')
    ap.add_argument('--walls', default='tmp/wall_geometry.csv',
                    help='per-wall landmark trace; graded independently of layout_trace, which the prediction early-exit starves after publication')
    a = ap.parse_args()

    lx, ly = a.gt
    gt = Polygon([(-lx / 2, -ly / 2), (lx / 2, -ly / 2), (lx / 2, ly / 2), (-lx / 2, ly / 2)])
    rows = load_layout(a.layout, a.since_ms)
    if not rows:
        sys.exit(f'no layout rows in {a.layout}')
    t0 = rows[0]['ts']
    print(f'GROUND TRUTH  {lx:.2f} x {ly:.2f} m rectangle, area {lx * ly:.2f} m^2, 4 corners at 90 deg')
    print(f'layout rows   {len(rows)}  ({(rows[-1]["ts"] - t0) / 1000:.1f} s)\n')

    first_closed = next((r for r in rows if r['closed']), None)
    first_pub = next((r for r in rows if r['publishable']), None)
    print('TIMELINE')
    print(f'  first closed      : {"never" if not first_closed else f"{(first_closed["ts"] - t0) / 1000:.1f} s (frame {first_closed["frame"]})"}')
    print(f'  first publishable : {"never" if not first_pub else f"{(first_pub["ts"] - t0) / 1000:.1f} s (frame {first_pub["frame"]})"}')
    if first_pub:
        after = [r for r in rows if r['ts'] >= first_pub['ts']]
        print(f'  publishable after first: {100 * sum(r["publishable"] for r in after) / len(after):.1f}% of {len(after)} rows')
    print()

    def show(label, row):
        g = grade_layout(row, gt)
        if g is None:
            print(f'{label}: fewer than 3 vertices'); return
        print(f'{label}  frame {row["frame"]}  t={(row["ts"] - t0) / 1000:.1f} s  closed={row["closed"]} publishable={row["publishable"]}')
        print(f'  vertices {g["n"]} (kinks <{KINK_DEG:.0f} deg: {g["kinks"]})   area {g["area"]:.2f} m^2 (gt {lx * ly:.2f})   '
              f'best-aligned IoU {g["iou"]:.4f}')
        print(f'  turns (deg): ' + ' '.join(f'{t:+.1f}' for t in g['turns']))
        print(f'  sides (m)  : ' + ' '.join(f'{s:.3f}' for s in g['sides']))
        errs = g['corners']
        print(f'  corner error after alignment (m) / published sigma / error in sigmas:')
        for i, (e, s) in enumerate(errs):
            z = e / s if s and s > 0 and not math.isnan(s) else float('nan')
            flag = '  <-- kink' if abs(g['turns'][i]) < KINK_DEG else ''
            print(f'    v{i}: {e:.3f}  sigma {s:.3f}  z {z:5.2f}{flag}')
        real = [(e, s) for i, (e, s) in enumerate(errs) if abs(g['turns'][i]) >= KINK_DEG]
        if real:
            cov1 = sum(1 for e, s in real if s > 0 and e <= s) / len(real)
            cov2 = sum(1 for e, s in real if s > 0 and e <= 2 * s) / len(real)
            print(f'  real corners: max err {max(e for e, _ in real):.3f} m   within 1 sigma {100 * cov1:.0f}%   within 2 sigma {100 * cov2:.0f}%')
        print()

    if first_pub:
        show('AT FIRST PUBLISH', first_pub)
    show('FINAL', rows[-1])

    tail = [r for r in rows if len(r['verts']) == len(rows[-1]['verts'])][-50:]
    if len(tail) >= 2:
        drift = max(max(math.dist(p, q) for p, q in zip(r['verts'], tail[-1]['verts'])) for r in tail)
        print(f'STABILITY  max vertex drift over the last {len(tail)} same-topology rows: {drift:.3f} m\n')

    if os.path.exists(a.walls):
        tl = walls_timeline(a.walls, a.since_ms)
        if tl:
            print('WALL LANDMARKS vs GROUND TRUTH  (tmp/wall_geometry.csv, every cycle; polygon = neighbouring walls intersected)')
            print(f'  {"t (s)":>7} {"walls":>5} {"IoU":>6} {"area":>6}  sides (m)                     worst side err  min frames_seen  max sigma_d')
            tw0 = tl[0][0]
            step = max(1, len(tl) // 25)
            picks = list(range(0, len(tl), step))
            if picks[-1] != len(tl) - 1:
                picks.append(len(tl) - 1)
            for i in picks:
                t, walls = tl[i]
                v = polygon_from_walls(walls)
                if v is None:
                    print(f'  {(t - tw0) / 1000:7.1f} {len(walls):5d}   (not a convex wall set)')
                    continue
                poly = Polygon(v)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                iou, _, _ = best_alignment(poly, gt)
                sd = sides(v)
                worst = max(min(abs(x - lx), abs(x - ly)) for x in sd) if len(sd) == 4 else float('nan')
                print(f'  {(t - tw0) / 1000:7.1f} {len(walls):5d} {iou:6.4f} {poly.area:6.2f}  '
                      + ' '.join(f'{x:6.3f}' for x in sd).ljust(29)
                      + f' {100 * worst:9.1f} cm  {min(w[3] for w in walls.values()):15d}  {max(w[2] for w in walls.values()):11.4f}')
            print()

    pose = grade_pose(a.gt_error, a.since_ms)
    if pose is None:
        print('POSE  no usable gt_error.csv rows')
        return
    print(f'POSE vs GROUND TRUTH  ({pose["n"]} rows, {pose["jumps"]} frame change(s) detected)')
    print('  per segment, after fitting est = R*gt + t (frames differ by design):')
    for s in pose['segs']:
        print(f'    {(s["t0"] - t0) / 1000:7.1f}..{(s["t1"] - t0) / 1000:7.1f} s  n={s["n"]:5d}  travel {s["travel"]:5.1f} m  '
              f'pos RMS {100 * s["rms"]:.1f} cm  p95 {100 * s["p95"]:.1f} cm  yaw-offset sd {s["yaw_sd_deg"]:.2f} deg')
        print(f'{"":22s}20 s windows: n={s["win_n"]}  median RMS {100 * s["win_med"]:.1f} cm  worst {100 * s["win_max"]:.1f} cm')


if __name__ == '__main__':
    main()
