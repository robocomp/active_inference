#!/usr/bin/env python3
"""anchor_vs_truth.py -- how far door_concept's interaction anchor (door_N@robot) was from the TRUE opening, per frame.

Inputs (a run directory of pingpong_tour.sh): anchor_log.csv (door_concept, AnchorLog = true: t_ms, x, y, yaw, sx, sy, syaw, ...
= the opening's pose in the ROBOT frame, +y forward, +x side), gt_pose.tsv (Webots: t_ms, x, y, theta), and the world file
(DOOR_1's centre and wall direction, adoption_mock's parser). The anchor's axes: x along the wall, y the wall normal.
Output, per crossing: the anchor's error along the wall (lateral -- what puts a body on a jamb), across it (depth) and in yaw,
against the robot's distance to the plane; and the error at the controller's stall events (controller/stall_events.csv).
The ground truth's heading convention is INFERRED (the direction of travel while driving forward), not assumed.
Usage: anchor_vs_truth.py <run_dir> [--anchor-log PATH] [--stalls PATH]
"""
import argparse, bisect, csv, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adoption_mock as am


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def load_gt(path):
    rows = [(float(r[0]), float(r[1]), float(r[2]), float(r[3])) for r in csv.reader(open(path), delimiter="\t") if r and r[0][0].isdigit()]
    return rows


def heading_offset(gt):
    """forward direction = theta + offset; offset from the travel direction over moving stretches (median, two candidates)"""
    d = []
    for (t0, x0, y0, a0), (t1, x1, y1, a1) in zip(gt, gt[1:]):
        dx, dy = x1 - x0, y1 - y0
        if math.hypot(dx, dy) > 0.03 and abs(wrap(a1 - a0)) < 0.05:
            d.append(wrap(math.atan2(dy, dx) - a0))
    if not d: return 0.0
    # forward driving dominates; fold to the nearest quarter turn and take the median
    q = [round(v / (math.pi / 2)) % 4 for v in d]
    k = max(range(4), key=q.count)
    return k * math.pi / 2


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--anchor-log"); ap.add_argument("--stalls")
    a = ap.parse_args()
    alog = a.anchor_log or os.path.join(a.run, "anchor_log.csv")
    gt = load_gt(os.path.join(a.run, "gt_pose.tsv"))
    tg = [r[0] for r in gt]
    off = heading_offset(gt)
    W = am.parse_world(); d1 = W.doors["DOOR_1"]
    c, tan = np.asarray(d1["centre"], float), np.asarray(d1["tangent"], float)
    tan = tan / np.linalg.norm(tan)
    print(f"{a.run}: DOOR_1 centre ({c[0]:.2f},{c[1]:.2f}); ground-truth forward = theta {math.degrees(off):+.0f} deg (inferred)")
    rows = list(csv.DictReader(open(alog)))
    out = []
    for r in rows:
        t = float(r["t_ms"]); i = bisect.bisect_left(tg, t)
        if i <= 0 or i >= len(gt): continue
        _, x, y, th = gt[i]
        fwd = th + off                      # robot +y
        side = fwd - math.pi / 2            # robot +x
        R = np.array([[math.cos(side), -math.sin(side)], [math.sin(side), math.cos(side)]])   # robot -> world
        # the anchor's centre in the world, from its robot-frame pose
        ca = np.array([x, y]) + R @ np.array([float(r["x"]), float(r["y"])])
        ya = side + float(r["yaw"])        # its x-axis (along the wall) in the world
        ua = np.array([math.cos(ya), math.sin(ya)])
        if np.dot(ua, tan) < 0: ua = -ua; ya += math.pi
        e = ca - c
        lat, dep = float(np.dot(e, tan)), float(np.dot(e, np.array([-tan[1], tan[0]])))
        yaw = math.degrees(wrap(math.atan2(ua[1], ua[0]) - math.atan2(tan[1], tan[0])))
        s_robot = float(np.dot(np.array([x, y]) - c, np.array([-tan[1], tan[0]])))
        out.append((t, s_robot, lat, dep, yaw, float(r["sx"]), float(r.get("p_view", "nan") or "nan")))
    if not out:
        print("  no anchor rows inside the ground-truth span"); return
    arr = np.array([(o[2], o[3], o[4]) for o in out])
    print(f"  {len(out)} anchor rows: |lateral| median {np.median(abs(arr[:,0])):.3f} m, 90% {np.percentile(abs(arr[:,0]),90):.3f}, "
          f"max {abs(arr[:,0]).max():.3f}; |depth| median {np.median(abs(arr[:,1])):.3f}; |yaw| median {np.median(abs(arr[:,2])):.1f} deg")
    print("  by the robot's distance to the plane (m): lateral error mean / max (m), published sigma_x (m)")
    for lo, hi in ((-3, -1.5), (-1.5, -0.75), (-0.75, 0), (0, 0.75), (0.75, 3)):
        sel = [o for o in out if lo <= o[1] < hi]
        if sel:
            l = np.array([o[2] for o in sel]); sx = np.array([o[5] for o in sel])
            print(f"    s in [{lo:+.2f},{hi:+.2f}): n {len(sel):4d}  lateral {l.mean():+.3f} / {abs(l).max():.3f}  sigma_x {np.median(sx):.3f}")
    stalls = a.stalls or "/home/pbustos/robocomp/components/active_inference/controller/stall_events.csv"
    if os.path.exists(stalls):
        ta = [o[0] for o in out]
        for r in csv.DictReader(open(stalls)):
            t = float(r["t_ms"])
            if not (ta[0] - 2000 <= t <= ta[-1] + 2000): continue
            j = min(range(len(ta)), key=lambda k: abs(ta[k] - t))
            o = out[j]
            print(f"  stall {r['verdict']:15s} t {t:.0f}: anchor lateral {o[2]:+.3f} m depth {o[3]:+.3f} yaw {o[4]:+.1f} deg "
                  f"(robot {o[1]:+.2f} m from the plane, nearest anchor row {abs(ta[j]-t):.0f} ms away)")


if __name__ == "__main__":
    main()
