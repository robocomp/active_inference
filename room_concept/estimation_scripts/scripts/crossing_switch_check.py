#!/usr/bin/env python3
"""crossing_switch_check.py -- offline check of spec 2026-10-31-room-crossing-v4 step 4: "which room explains the scan
better" decides the switch.  Question: on the robot's RECORDED crossings of DOOR_1, does the {old, rival} belief switch
after the body is through the opening and within ~1 m of it, and never while the robot stands at the open door?

CAVEAT: a MODEL of the comparison, not room_concept's code (no SDF fit, no Adam): the scans are SYNTHESISED from the
Webots world (threerooms-piso.wbt, via adoption_mock's parser/raycaster) along the ground-truth path (gt_pose.tsv).
What it can say: whether the decision rule itself is sound on this geometry.  What it cannot: how the agent's own
fit / pose noise behaves.

Models
  old    = the room left, its outline FROZEN (true outline, doors sealed: build_memory_room).
  rival  = remembered: the other room's true outline moved by a recall error (--recall-err m, --recall-rot deg);
           new:        a thin estimate: the scan endpoints seen so far on the far side of the door plane (a point
                       cloud; distance to the nearest), i.e. what the proto has before any survey.
  point likelihood under a room: p = (1-eps) N(d; 0, sigma) + eps * U   (U: an unexplained return; the same floor for
  both rooms, so a return neither room explains carries ~0 nats -- the "neutral" requirement).
  belief over {old, rival}: sticky HMM, hazard h per scan; the switch = P(rival) > 1/2.
Usage: python3 crossing_switch_check.py [--run pingpong29] [--mode remembered|new|both]
"""
import argparse, csv, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adoption_mock as am

CROSS = "/home/pbustos/robocomp/components/active_inference/room_concept/estimation_scripts/crossing"
DOOR1_Y = 9.3796
R_BODY = 0.28


def seg_dist(P, A, B):
    """distance from points P (N,2) to the polygon edges A->B (M,2): (N,)"""
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(1), 1e-12)
    t = np.clip(((P[:, None, :] - A[None]) * AB[None]).sum(2) / L2[None], 0, 1)
    Q = A[None] + t[..., None] * AB[None]
    return np.sqrt(((P[:, None, :] - Q) ** 2).sum(2)).min(1)


def poly_edges(poly):
    P = np.asarray(poly, float)
    return P, np.roll(P, -1, 0)


def loglik(d, sigma, eps, U):
    return np.log((1 - eps) * np.exp(-0.5 * (d / sigma) ** 2) / (math.sqrt(2 * math.pi) * sigma) + eps * U)


def crossings(run, cx):
    rows = [(float(r[0]), float(r[1]), float(r[2]), float(r[3])) for r in csv.reader(open(f"{CROSS}/{run}/gt_pose.tsv"), delimiter="\t")
            if r and r[0][0].isdigit()]
    out, i = [], 1
    while i < len(rows):
        if (rows[i - 1][2] - DOOR1_Y) * (rows[i][2] - DOOR1_Y) < 0 and abs(rows[i][1] - cx) < 1.0:
            k = i
            j0 = k
            while j0 > 0 and rows[k][0] - rows[j0][0] < 25000: j0 -= 1     # 25 s before the plane (the wait at the door)
            j1 = k
            while j1 < len(rows) - 1 and rows[j1][0] - rows[k][0] < 12000: j1 += 1
            out.append((rows[j0:j1], +1 if rows[i][2] > rows[i - 1][2] else -1))
            i = j1
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pingpong29"); ap.add_argument("--mode", default="both")
    ap.add_argument("--sigma", type=float, default=0.05); ap.add_argument("--eps", type=float, default=0.2)
    ap.add_argument("--hazard", type=float, default=1e-3); ap.add_argument("--pose-err", type=float, default=0.05)
    ap.add_argument("--recall-err", type=float, default=0.15); ap.add_argument("--recall-rot", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=7); ap.add_argument("--beams", type=int, default=180); ap.add_argument("--every", type=int, default=2)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    W = am.parse_world()
    d1 = W.doors["DOOR_1"]
    # seal every door but DOOR_1 for the scans (the tour keeps them shut)
    segs = list(W.segments)
    for d in W.doors.values():
        if d is d1: continue
        c, t = d["centre"], d["tangent"]
        segs.append((c - d["jamb_in"] * t, c + d["jamb_in"] * t))
    Wscan = am.World(); Wscan.segments = segs
    room1 = np.asarray(am.build_memory_room(W, seed_xy=(d1["centre"][0], DOOR1_Y - 1.5))[0])
    room2 = np.asarray(am.build_memory_room(W, seed_xy=(d1["centre"][0], DOOR1_Y + 1.5))[0])
    U = 1.0 / 12.0   # uniform over ~12 m of range
    print(f"{a.run}: DOOR_1 at ({d1['centre'][0]:.2f},{d1['centre'][1]:.2f}); room_1 {am.polygon_area(room1):.1f} m2, room_2 {am.polygon_area(room2):.1f} m2")
    modes = ["remembered", "new"] if a.mode == "both" else [a.mode]
    for ci, (path, sgn) in enumerate(crossings(a.run, d1['centre'][0])):
        old, other = (room1, room2) if sgn > 0 else (room2, room1)
        for mode in modes:
            th = math.radians(a.recall_rot) * rng.choice([-1, 1]); off = rng.normal(0, a.recall_err, 2)
            R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
            riv_poly = (np.asarray(other) - d1["centre"]) @ R.T + d1["centre"] + off
            oA, oB = poly_edges(old); rA, rB = poly_edges(riv_poly)
            cloud = np.zeros((0, 2))
            b = np.array([1 - 1e-3, 1e-3]); switch = None; pre_flip = False; s_min = 1e9
            for k in range(0, len(path), a.every):
                t, x, y, yaw = path[k]
                s = sgn * (y - DOOR1_Y)       # body centre past the door plane, along the crossing direction
                pose = np.array([x, y, yaw])
                pts_r = am.lidar_scan(Wscan, pose, rng, n_beams=a.beams, sigma=0.02) if hasattr(am, "lidar_scan") else None
                pts_r = np.asarray(pts_r, float)[:, :2]
                pe = pose + np.array([rng.normal(0, a.pose_err), rng.normal(0, a.pose_err), rng.normal(0, math.radians(1))])
                c_, s_ = math.cos(pe[2]), math.sin(pe[2])
                P = pts_r @ np.array([[c_, s_], [-s_, c_]]) + pe[:2]
                lo = loglik(seg_dist(P, oA, oB), a.sigma, a.eps, U)
                if mode == "remembered":
                    lr = loglik(seg_dist(P, rA, rB), a.sigma, a.eps, U)
                else:
                    if len(cloud):
                        dd = np.sqrt(((P[:, None, :] - cloud[None]) ** 2).sum(2)).min(1)
                        lr = loglik(dd, a.sigma, a.eps, U)
                    else:
                        lr = np.full(len(P), math.log(a.eps * U))
                    far = P[(sgn * (P[:, 1] - DOOR1_Y)) > 0.15]
                    cloud = np.vstack([cloud, far])[-3000:]
                llr = float(np.sum(lr - lo))
                h = a.hazard
                pr = np.array([b[0] * (1 - h) + b[1] * h, b[0] * h + b[1] * (1 - h)])
                m = max(0.0, llr); post = pr * np.array([math.exp(-m), math.exp(llr - m)]); b = post / post.sum()
                if b[1] > 0.5 and switch is None:
                    switch = s
                    if s < R_BODY: pre_flip = True
                if b[1] > 0.5 and s < 0: pre_flip = True
            verdict = "PASS" if switch is not None and R_BODY <= switch <= 1.3 and not pre_flip else "FAIL"
            print(f"  crossing {ci + 1} ({'1->2' if sgn > 0 else '2->1'}) {mode:10s}: switch at body {switch if switch is None else round(switch, 2)} m past the plane"
                  f" (want {R_BODY:.2f}..1.3){' ; flipped BEFORE the body was through' if pre_flip else ''} -- {verdict}")


if __name__ == "__main__":
    main()
