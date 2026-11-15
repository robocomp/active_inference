#!/usr/bin/env python3
"""Ray-cast half of the carve replay (lifted from docs/superpowers/results/2026-11-15-carve-replay.py).

For every "drop cycle" of a recorded run (a cycle where a real object's existence L fell while LiDAR absence was
non-zero, every other one), cast the simulated Helios sweep from the Webots ground-truth robot pose against the
true scene boxes and write the first-hit points. The SCORING is done in C++ by carve_replay.cpp, with the agent's
own carve_box / carve_parts.

Usage: carve_replay_rays.py RUN_DIR SCENE_CSV OUT.bin [--phantoms] [--max-cycles N]

Record (little endian), WORLD frame:
  u32 n_name, name bytes, f32 cx cy yaw w h H std_cx std_cy, f32 origin xyz, u32 n, n * f32 xyz
--phantoms adds, per replayed cycle, two synthetic nodes with the real node's logged box size:
  phantom_floor_<node>  the box displaced 1.0 m along the first of +x -x +y -y whose footprint overlaps no GT box (the distance grows past 1.0 m if all four overlap)
  phantom_wall_<node>   the box centred on the nearest wall's inner face
"""
import argparse, collections, csv, math, struct, sys
import numpy as np

H_ORIGIN = 1.075 + 0.0325                         # Helios height above the floor
BEHIND = 0.155                                    # Helios is this far behind the base along the robot's facing
ELEV = np.deg2rad(np.linspace(-15.5, 54.5, 32))   # inverted H32F70 rings
AZ = np.deg2rad(np.arange(0, 360, 0.4))
DIRS = np.array([[math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)] for e in ELEV for a in AZ])
MAX_RANGE = 30.0

# Measured on the Webots world / the objects_live6 fits (see the results doc "Method"): node centre in the world,
# and the fitted centre in the room frame, used to place the bedroom (rotation 0) and the bathroom (-pi/2).
GTC = {'bed_10': (-5.18246, 12.3911), 'nightstand_6': (-4.0035, 13.216), 'nightstand_7': (-6.3615, 13.216),
       'toilet_6': (-0.874478, 12.8652)}
FIT = {'bed_10': (-0.36, 3.04), 'nightstand_6': (0.87, 3.89), 'nightstand_7': (-1.60, 3.91), 'toilet_6': (-2.32, 1.10)}
ROOM_OF = {'bed': 'bedroom', 'nightstand': 'bedroom', 'toilet': 'bathroom'}


def load_scene(path):
    rows = list(csv.DictReader(open(path)))
    lab = [r['label'] for r in rows]
    B = np.array([[float(r[k]) for k in ('xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax')] for r in rows])
    return B, lab


def cast(B, o, d):
    """first hit of rays o + t d (d unit, N x 3) against boxes, floor z=0, ceiling z=3 -> t"""
    tbest = np.full(d.shape[0], MAX_RANGE)
    with np.errstate(divide='ignore', invalid='ignore'):
        inv = 1.0 / d
        for k in range(B.shape[0]):
            t1 = (B[k, [0, 2, 4]] - o) * inv; t2 = (B[k, [1, 3, 5]] - o) * inv
            tn = np.nanmax(np.minimum(t1, t2), axis=1); tf = np.nanmin(np.maximum(t1, t2), axis=1)
            hit = (tn <= tf) & (tf > 0.09) & (tn < tbest)
            tn = np.where(tn < 0.09, tf, tn)
            tbest = np.where(hit, tn, tbest)
        for zp in (0.0, 3.0):
            t = (zp - o[2]) / d[:, 2]
            hit = (t > 0.09) & (t < tbest)
            tbest = np.where(hit, t, tbest)
    return tbest


def room_transforms():
    tb = np.mean([np.subtract(GTC[k], FIT[k]) for k in ('bed_10', 'nightstand_6', 'nightstand_7')], axis=0)
    c90 = -math.pi / 2
    R3 = np.array([[math.cos(c90), -math.sin(c90)], [math.sin(c90), math.cos(c90)]])
    return {'bedroom': (np.eye(2), tb, 0.0),
            'bathroom': (R3, np.array(GTC['toilet_6']) - R3 @ np.array(FIT['toilet_6']), c90)}


def robot_world(gt, ts):
    if ts < gt[0, 0] or ts > gt[-1, 0]: return None
    k = max(1, min(int(np.searchsorted(gt[:, 0], ts)), len(gt) - 1))
    a, b = gt[k - 1], gt[k]
    if b[0] - a[0] > 2500: return None
    f = (ts - a[0]) / (b[0] - a[0]); dyaw = ((b[3] - a[3] + math.pi) % (2 * math.pi)) - math.pi
    return (a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])), a[3] + f * dyaw


def footprint_aabb(cx, cy, yaw, w, h):
    c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
    ex, ey = 0.5 * (c * w + s * h), 0.5 * (s * w + c * h)
    return cx - ex, cx + ex, cy - ey, cy + ey


def overlaps(B, a):
    return bool(np.any((a[0] < B[:, 1]) & (a[1] > B[:, 0]) & (a[2] < B[:, 3]) & (a[3] > B[:, 2])))


def floor_displacement(B, cx, cy, yaw, w, h):
    """first of +x -x +y -y whose displaced footprint overlaps no GT box. 1.0 m is the ruling, but a box longer than
    1 m (the bed) still overlaps ITSELF at 1.0 m, so the distance grows by 0.25 m until a direction is free."""
    d = 1.0
    while d < 6.0:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if not overlaps(B, footprint_aabb(cx + dx * d, cy + dy * d, yaw, w, h)): return (dx * d, dy * d)
        d += 0.25
    return (0.0, 0.0)


def wall_position(B, lab, cx, cy, yaw, w, h):
    """centre on the inner face of the nearest wall (the face turned toward the node) whose phantom overlaps no
    non-wall GT box: the nearest wall of a nightstand is the one it stands against, and a phantom centred there
    would sit half on the real object, which tests nothing"""
    solid = np.array([l != 'wall' for l in lab])
    cands = []
    for k, l in enumerate(lab):
        if l != 'wall': continue
        x0, x1, y0, y1 = B[k, :4]
        px, py = min(max(cx, x0), x1), min(max(cy, y0), y1)
        if (x1 - x0) < (y1 - y0):                   # thin in x: wall runs along y, the face is an x plane
            pos = (x0 if cx < 0.5 * (x0 + x1) else x1, py)
        else:
            pos = (px, y0 if cy < 0.5 * (y0 + y1) else y1)
        cands.append((math.hypot(cx - px, cy - py), pos))
    cands.sort(key=lambda c: c[0])
    for _, pos in cands:
        if not overlaps(B[solid], footprint_aabb(pos[0], pos[1], yaw, w, h)): return pos
    return None


def pack(name, box, std, origin, pts):
    nb = name.encode()
    return (struct.pack('<I', len(nb)) + nb + struct.pack('<8f', *box, *std) + struct.pack('<3f', *origin)
            + struct.pack('<I', len(pts)) + np.ascontiguousarray(pts, dtype='<f4').tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run'); ap.add_argument('scene'); ap.add_argument('out')
    ap.add_argument('--phantoms', action='store_true')
    ap.add_argument('--max-cycles', type=int, default=0, help='replay at most N drop cycles per node (0 = all)')
    a = ap.parse_args()
    B, lab = load_scene(a.scene)
    gt = np.array([(int(r['t_ms']), float(r['x']), float(r['y']), float(r['yaw']))
                   for r in csv.DictReader(open(f'{a.run}/gt_pose.tsv'), delimiter='\t')])
    XF = room_transforms()
    out = open(a.out, 'wb'); total = collections.Counter(); wall_warned = set()
    for cls, room in ROOM_OF.items():
        R, t, dth = XF[room]
        by = collections.defaultdict(list)
        for r in csv.DictReader(open(f'{a.run}/{cls}_ai2_log.csv')): by[r['node']].append(r)
        for node, rs in by.items():
            if len(rs) < 200: continue
            prev = None; n = 0; fdisp = None; cand = []
            for i, r in enumerate(rs):
                try:
                    L = float(r['ex_L']); lfe = float(r['ex_lfree_eff'])
                except (ValueError, KeyError):
                    prev = None; continue
                drop = prev is not None and L < prev - 1e-4 and lfe > 0; prev = L
                if drop and i % 2 == 0: cand.append(i)
            if a.max_cycles and len(cand) > a.max_cycles:          # N cycles EVENLY across the run
                cand = [cand[int(j * len(cand) / a.max_cycles)] for j in range(a.max_cycles)]
            for i in cand:
                r = rs[i]
                try:
                    ts = int(float(r['pkt_ts']))
                    cx, cy, yaw, w, h, H = (float(r[k]) for k in ('cx', 'cy', 'yaw', 'w', 'h', 'H'))
                    sx, sy = float(r['std_cx']), float(r['std_cy'])
                except (ValueError, KeyError):
                    continue
                rp = robot_world(gt, ts)
                if rp is None: continue
                (px, py), th = rp
                O = np.array([px - BEHIND * -math.sin(th), py - BEHIND * math.cos(th), H_ORIGIN])
                tt = cast(B, O, DIRS)
                P = (O + DIRS * tt[:, None])[tt < MAX_RANGE - 1e-6]
                cw = R @ np.array([cx, cy]) + t; yw = yaw + dth
                std = (sx, sy)
                out.write(pack(node, (cw[0], cw[1], yw, w, h, H), std, O, P)); total[node] += 1
                if a.phantoms:
                    if fdisp is None: fdisp = floor_displacement(B, cw[0], cw[1], yw, w, h) or (0.0, 0.0)
                    out.write(pack('phantom_floor_' + node, (cw[0] + fdisp[0], cw[1] + fdisp[1], yw, w, h, H), std, O, P))
                    wp = wall_position(B, lab, cw[0], cw[1], yw, w, h)
                    if wp is None:
                        if node not in wall_warned:
                            print(f'WARNING {node}: every wall candidate overlaps a non-wall GT box, NO wall phantom', file=sys.stderr)
                            wall_warned.add(node)
                    else:
                        if node not in wall_warned:
                            print(f'phantom_wall_{node}: centred on a wall inner face at ({wp[0]:.3f}, {wp[1]:.3f})', file=sys.stderr)
                            wall_warned.add(node)
                        out.write(pack('phantom_wall_' + node, (wp[0], wp[1], yw, w, h, H), std, O, P))
                        total['phantom_wall_' + node] += 1
                    total['phantom_floor_' + node] += 1
                n += 1
    out.close()
    for k, v in total.items(): print(f'{k}: {v} cycles', file=sys.stderr)


main()
