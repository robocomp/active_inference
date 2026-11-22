#!/usr/bin/env python3
"""Published room frame -> Webots world, for one tour run folder (2026-11-22).

Same path as c3a_objects.py's to_world (gauge branch): a published (x, y) at time t is taken to room_concept's INTERNAL
frame by the logged gauge (proto_room.csv gauge_x/y/theta = T_pub_int, room_scene_graph.cpp:1363), then to the world by
the rigid fit of est -> gt over the visit nearest t (gt_error.csv, visits from legs.tsv, trimmed by 3 s at both ends).
    to_world = make_to_world(run);  to_world(x, y, t_ms) -> (wx, wy) or None
"""
import bisect, csv, math, os
import numpy as np


def rigid_fit(A, B):
    A, B = np.asarray(A), np.asarray(B)
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    th = math.atan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
    Rm = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    return th, cb - Rm @ ca


def make_to_world(run):
    legs = {int(r['leg']): (int(r['t0_ms']), int(r['t1_ms'])) for r in csv.DictReader(open(f'{run}/legs.tsv'), delimiter='\t')}
    est = [(int(r['ts_ms']), float(r['est_x']), float(r['est_y']), float(r['gt_x_world']), float(r['gt_y_world']))
           for r in csv.DictReader(open(f'{run}/gt_error.csv'))]
    edges = [est[0][0]] + [t for k in sorted(legs) for t in legs[k]] + [est[-1][0]]
    fits = []
    for a, b in zip(edges[0::2], edges[1::2]):           # visits: start..leg1.t0, leg1.t1..leg2.t0, ...
        sel = [e for e in est if a + 3000 <= e[0] <= b - 3000]
        if len(sel) >= 10: fits.append((a, b) + rigid_fit([e[1:3] for e in sel], [e[3:5] for e in sel]))
    G = []
    for r in csv.DictReader(open(f'{run}/proto_room.csv')):
        try: G.append((int(r['ts_ms']), float(r['gauge_x']), float(r['gauge_y']), float(r['gauge_theta'])))
        except (KeyError, ValueError): pass
    Gts = [g[0] for g in G]

    def to_world(x, y, t_ms):
        if not fits or not G: return None
        _, g0, g1, gth = G[max(bisect.bisect_right(Gts, t_ms) - 1, 0)]
        dx, dy = x - g0, y - g1
        mx, my = math.cos(gth) * dx + math.sin(gth) * dy, -math.sin(gth) * dx + math.cos(gth) * dy
        a, b, th, t = min(fits, key=lambda f: 0 if f[0] <= t_ms <= f[1] else min(abs(t_ms - f[0]), abs(t_ms - f[1])))
        return (math.cos(th) * mx - math.sin(th) * my + t[0], math.sin(th) * mx + math.cos(th) * my + t[1])
    return to_world
