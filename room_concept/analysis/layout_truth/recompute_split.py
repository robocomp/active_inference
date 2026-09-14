#!/usr/bin/env python3
"""Where does the paper's 9.5 cm "map offset" actually come from?

The §V-E decomposition is reproduced verbatim (the original was an ad-hoc heredoc; see the
report), then ATTRIBUTED: every model corner the probe uses is placed back in the room
frame, the detected corners are un-projected into the room frame through the same per-frame
pose, and the per-corner bias is compared against that corner's KNOWN map error measured
against the exact Webots collision/render geometry.

Run: tmp/corner_probe.csv as it stands on disk.  NOTE this is NOT the paper's run (see the
report): 129657 rows / 13167 frames here vs the cited 126008 candidates / 8925 frames.
"""
import numpy as np, json, math, re, os
from collections import defaultdict

S = "/tmp/claude-1000/-home-pbustos-robocomp-components-active-inference-room-concept/c1e48398-243f-4bda-b915-064cb4be9944/scratchpad"
A = np.load(S + "/probe.npy"); H = json.loads(open(S + "/probe_cols.json").read())
c = {n: i for i, n in enumerate(H)}
EXACT = np.load(S + "/exact.npy")

# ---- the model the agent actually used: apartamento_layout.svg through SvgRoomLoader ----
txt = open('/home/pbustos/robocomp/components/active_inference/layouts/apartamento_layout.svg').read()
svg = np.array([[float(v) for v in t.split(",")]
                for t in re.search(r'points="([^"]+)"', txt, re.S).group(1).split()])
loaded = svg.copy(); loaded[:, 0] *= -1; loaded = loaded[::-1]
MODEL_W = loaded                                   # webots world frame
mn, mx = loaded.min(0), loaded.max(0)
CENTRE = (mn + mx) / 2
MODEL = loaded - CENTRE                            # RecenterRoomPolygon=true: the agent's room frame

# ---- the truth each model corner should have had -------------------------------------
# Non-pillar corners: the wall mesh (exact, verified).  Pillar-bay corners: the pillar
# Solids of piso.wbt.  Webots' Lidar is an OpenGL depth render, so it sees the child Shape
# geometry (Box 0.25 x 0.4 x 3), not the boundingObject (Box 0.4 x 0.6 x 3).  Both
# hypotheses are carried and the DATA is asked to decide between them.
PILLARS = {"Pillar": dict(t=(-8.31, 1.07), shape=(0.25, 0.4), bound=(0.4, 0.6)),
           "Pillar(1)": dict(t=(-0.310883, 1.08), shape=(0.25, 0.4), bound=(0.4, 0.6))}
def box_corners(t, s):
    return np.array([[t[0] - s[0] / 2, t[1] - s[1] / 2], [t[0] + s[0] / 2, t[1] - s[1] / 2],
                     [t[0] + s[0] / 2, t[1] + s[1] / 2], [t[0] - s[0] / 2, t[1] + s[1] / 2]])
PIL_SHAPE = np.vstack([box_corners(p["t"], p["shape"]) for p in PILLARS.values()])
PIL_BOUND = np.vstack([box_corners(p["t"], p["bound"]) for p in PILLARS.values()])
PILLAR_IDX = set([10, 11, 12, 13, 27, 28, 29, 30])      # the two bays in the loaded order

def nearest(P, q): return float(np.linalg.norm(P - q, axis=1).min())

def map_error(k):
    """Distance from model corner k to the nearest TRUE feature, and which one."""
    w = MODEL_W[k]
    if k in PILLAR_IDX:
        return nearest(PIL_SHAPE, w), nearest(PIL_BOUND, w), "pillar"
    return nearest(EXACT, w), nearest(EXACT, w), "wall"

# ---- per-frame pose from the probe itself ---------------------------------------------
def kabsch(P, Q):
    cp, cq = P.mean(0), Q.mean(0); X, Y = P - cp, Q - cq
    U, _, Vt = np.linalg.svd(X.T @ Y); d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, d]) @ U.T
    return R, cq - R @ cp

fr = A[:, c['frame']].astype(int); mi = A[:, c['model_index']].astype(int)
pred = A[:, [c['pred_x'], c['pred_y']]]; det = A[:, [c['det_x'], c['det_y']]]
nu = A[:, [c['nu_x'], c['nu_y']]]
sdet = np.sqrt(np.maximum(0.0, (A[:, c['sdet_xx']] + A[:, c['sdet_yy']]) / 2))

order = np.argsort(fr, kind="stable")
frames = np.unique(fr)
pose = {}
det_room = np.full_like(det, np.nan)
for f in frames:
    m = fr == f
    if m.sum() < 3: continue
    R, t = kabsch(MODEL[mi[m]], pred[m])            # room -> robot
    if np.abs((MODEL[mi[m]] @ R.T + t) - pred[m]).max() > 1e-4: continue
    pose[f] = (R, t)
    det_room[m] = (det[m] - t) @ R                  # R^T (p - t)

# ---- 1. the paper's decomposition, verbatim -------------------------------------------
byf = defaultdict(dict)
for i in range(len(A)): byf[fr[i]][mi[i]] = (pred[i, 0], pred[i, 1])
fl = sorted(byf); still = set()
for a, b in zip(fl, fl[1:]):
    if b - a != 1: continue
    sh = set(byf[a]) & set(byf[b])
    if len(sh) < 3: continue
    P = np.array([byf[a][k] for k in sh]); Q = np.array([byf[b][k] for k in sh])
    R, _ = kabsch(P, Q)
    if abs(math.degrees(math.atan2(R[1, 0], R[0, 0]))) < 0.25 and \
       np.linalg.norm(Q.mean(0) - R @ P.mean(0)) < 0.005:
        still.add(b)
stretches, cur = [], []
for f in fl:
    if f in still: cur.append(f)
    else:
        if len(cur) >= 25: stretches.append(cur)
        cur = []
if len(cur) >= 25: stretches.append(cur)

clean = sdet < 0.08
rows_by_f = defaultdict(list)
for i in np.where(clean)[0]: rows_by_f[fr[i]].append(i)

pairs = []
for sid, st in enumerate(stretches):
    per = defaultdict(list)
    for f in st:
        for i in rows_by_f.get(f, []): per[mi[i]].append(i)
    for k, idxs in per.items():
        if len(idxs) < 25: continue
        N = nu[idxs]
        bias = float(np.linalg.norm(N.mean(0))); noise = float(np.linalg.norm(N.std(0)))
        # the same bias expressed in the ROOM frame (so it can be compared with a map error)
        dr = det_room[idxs]; dr = dr[~np.isnan(dr[:, 0])]
        room_bias = float(np.linalg.norm(dr.mean(0) - MODEL[k])) if len(dr) else np.nan
        room_pos = dr.mean(0) if len(dr) else np.array([np.nan, np.nan])
        pairs.append(dict(stretch=sid, k=k, n=len(idxs), bias=bias, noise=noise,
                          ratio=bias / max(1e-6, noise), room_bias=room_bias,
                          room_pos=room_pos))

b = np.array([p['bias'] for p in pairs]); nz = np.array([p['noise'] for p in pairs])
rt = np.array([p['ratio'] for p in pairs])
print(f"REPRODUCED on the CURRENT tmp/corner_probe.csv ({len(A)} rows, {len(frames)} frames)")
print(f"  {len(stretches)} parked stretches (>=25 consecutive still frames), {len(pairs)} corner-stretch pairs")
print(f"  ||mean nu||  REPEATS   median {np.median(b):.4f} m   p90 {np.percentile(b,90):.4f}")
print(f"  ||std nu||   SCATTERS  median {np.median(nz):.4f} m   p90 {np.percentile(nz,90):.4f}")
print(f"  ratio                  median {np.median(rt):.1f}x   bias>scatter in {100*(rt>1).mean():.0f}%")
print(f"  [paper cites 0.0952 / 0.0113 / 8.1x / 97% on a run that no longer exists]")

# ---- 2. attribution: bias per model corner vs that corner's map error ------------------
print("\n=== per model corner: does the repeatable bias match a KNOWN map error? ===")
print(f"{'idx':>4} {'model corner (webots)':>24} {'kind':>7} {'err_shape':>9} {'err_bnd':>8} "
      f"{'n_pairs':>7} {'med bias':>9} {'med room-bias':>13} {'detected corner (room->webots)':>31}")
tot = defaultdict(list)
for k in sorted(set(p['k'] for p in pairs)):
    ps = [p for p in pairs if p['k'] == k]
    es, eb, kind = map_error(k)
    rb = np.array([p['room_bias'] for p in ps]); rb = rb[~np.isnan(rb)]
    rp = np.array([p['room_pos'] for p in ps]); rp = rp[~np.isnan(rp[:, 0])]
    mean_w = rp.mean(0) + CENTRE if len(rp) else np.array([np.nan, np.nan])
    print(f"{k:4d} ({MODEL_W[k][0]:9.4f},{MODEL_W[k][1]:8.4f})      {kind:>7} {es:9.4f} {eb:8.4f} "
          f"{len(ps):7d} {np.median([p['bias'] for p in ps]):9.4f} "
          f"{(np.median(rb) if len(rb) else float('nan')):13.4f}   "
          f"({mean_w[0]:9.4f},{mean_w[1]:8.4f})")
    tot[kind].append((np.median([p['bias'] for p in ps]), len(ps)))

print("\n=== the split by corner KIND ===")
for kind in ("wall", "pillar"):
    ps = [p for p in pairs if map_error(p['k'])[2] == kind]
    if not ps: continue
    bb = np.array([p['bias'] for p in ps]); nn = np.array([p['noise'] for p in ps])
    print(f"  {kind:>7}: {len(ps):3d} corner-stretch pairs   median bias {np.median(bb):.4f} m   "
          f"median scatter {np.median(nn):.4f} m   ratio {np.median(bb)/np.median(nn):.1f}x")
ps_w = [p for p in pairs if map_error(p['k'])[2] == "wall"]
print(f"\n  => corners whose model geometry is EXACT to the micron still show a median "
      f"repeatable bias of {np.median([p['bias'] for p in ps_w]):.4f} m")

# ---- 3. which pillar hypothesis does the data support? --------------------------------
print("\n=== pillar: rendered Shape (0.25x0.40) or boundingObject (0.40x0.60)? ===")
for k in sorted(PILLAR_IDX & set(p['k'] for p in pairs)):
    ps = [p for p in pairs if p['k'] == k]
    rp = np.array([p['room_pos'] for p in ps]); rp = rp[~np.isnan(rp[:, 0])]
    if not len(rp): continue
    w = rp.mean(0) + CENTRE
    print(f"  idx {k:2d}: detected at ({w[0]:8.4f},{w[1]:7.4f})   "
          f"d(model)={np.linalg.norm(w-MODEL_W[k]):.4f}  "
          f"d(Shape corner)={nearest(PIL_SHAPE,w):.4f}  d(bound corner)={nearest(PIL_BOUND,w):.4f}")

# ---- 4. is the repeatable term VIEWPOINT-INVARIANT, as a map offset must be? -----------
# A map offset displaces the true corner: the detected corner, expressed in the ROOM frame,
# must then land in the SAME wrong place from every viewpoint.  A pose or detection bias
# does not.  So: per model corner, how much does the room-frame detected position move
# BETWEEN parked stretches, compared with how much it repeats WITHIN one?
print("\n=== is the repeatable term the same from every viewpoint? ===")
print(f"{'idx':>4} {'kind':>7} {'stretches':>9} {'within-stretch':>14} {'between-stretch':>15} {'verdict':>26}")
wn, bt = [], []
for k in sorted(set(p['k'] for p in pairs)):
    ps = [p for p in pairs if p['k'] == k and not np.isnan(p['room_pos'][0])]
    if len(ps) < 3: continue
    P = np.array([p['room_pos'] for p in ps])
    within = float(np.median([p['noise'] for p in ps]))          # scatter inside a stretch
    between = float(np.sqrt(((P - P.mean(0)) ** 2).sum(1).mean()))  # RMS spread across stretches
    kind = map_error(k)[2]
    wn.append(within); bt.append(between)
    v = "MOVES with viewpoint" if between > 3 * within else "stable"
    print(f"{k:4d} {kind:>7} {len(ps):9d} {within:14.4f} {between:15.4f} {v:>26}")
print(f"\n  median within-stretch scatter  {np.median(wn):.4f} m")
print(f"  median between-stretch spread  {np.median(bt):.4f} m   "
      f"({np.median(bt)/np.median(wn):.1f}x larger)")
print("  A map offset would make the between-stretch spread ~0.  It does not.")
