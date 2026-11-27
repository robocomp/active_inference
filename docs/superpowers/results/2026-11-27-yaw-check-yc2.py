#!/usr/bin/env python3
"""YC2 (2026-11-27): where does the staged wall's direction error live, in the room polygon or in the scorer's frame fit?
    python3 -I 2026-11-27-yaw-check-yc2.py [--boot B] [--csv OUT.csv] [--cache DIR] [--tours DIR]
Offline, measurement only. For each graded visit that YC1 could give a true wall (37), the stay's room->world rotation is measured a
SECOND way, from the stay's recorded LiDAR scans (scan_tape_*.bin, rc::tape, room_concept/src/scan_tape.h), WITHOUT using the robot's
heading to define the rotation against the truth: the scans are placed in the published (internal) frame by the tape's own estimated
pose, then a rigid transform (phi, t) of that frame is fitted so that the scan points lie on the TRUE Webots Wall boxes (point-to-line,
iterated, robust loss).  The scorer's rotation is the mean of (est heading - true heading) over the stay; this one is the rotation that
makes the scans fit the walls.  The two are compared; the staged edge direction error is recomputed in the independent frame.
The registration's own uncertainty: (1) a NULL TEST, the same fit with the scans placed by the exact Webots pose (rotation should be a
constant, the lidar mount), whose spread is the noise floor of a stay's registration; (2) a block bootstrap over the stay's frames.
Reuses YC1 (2026-11-27-yaw-check.py) for the visits, the staged edge and the scorer's frame; all numbers are read with float()/struct."""
import importlib.util, math, os, struct, sys
import numpy as np
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('yc1', os.path.join(HERE, '2026-11-27-yaw-check.py'))
yc1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(yc1)
F, G, R, PI, D = yc1.F, yc1.G, yc1.R, yc1.PI, yc1.D
wrap, fold = yc1.wrap, yc1.fold
from pathlib import Path
rng = np.random.default_rng(20261127)

# ---------------------------------------------------------------- the scan tape (scan_tape.h: "SCTP", u32 version, [u8 type][u32 n][payload])
def read_tape(path):
    b = open(path, 'rb').read()
    assert b[:4] == b'SCTP' and struct.unpack_from('<I', b, 4)[0] == 1
    o, ts, pose, pts = 8, [], [], []
    while o + 5 <= len(b):
        t = b[o]; n = struct.unpack_from('<I', b, o + 1)[0]; o += 5
        if o + n > len(b): break
        pl = b[o:o + n]; o += n
        if t != 2: continue
        ts.append(struct.unpack_from('<q', pl, 0)[0]); pose.append(struct.unpack_from('<3f', pl, 8))
        q = 8 + 12 + 36 + 12 + 36; q += 4 + struct.unpack_from('<I', pl, q)[0]
        npt = struct.unpack_from('<I', pl, q)[0]; q += 4
        pts.append(np.frombuffer(pl, dtype='<f4', count=2 * npt, offset=q).reshape(-1, 2).astype(float))
    return np.array(ts), np.array(pose), pts

# ---------------------------------------------------------------- the true walls: the four edges of every Webots Wall box, world frame
def wall_segments(wbt):
    import wbt_rooms as W
    walls, _, _ = W.parse_wbt(wbt)
    S = []
    for w in walls:
        sx, sy = w['size']; c, s = math.cos(w['ang']), math.sin(w['ang'])
        for (a, b) in (((-1, -1), (1, -1)), ((1, -1), (1, 1)), ((1, 1), (-1, 1)), ((-1, 1), (-1, -1))):
            def P(u): return (w['c'][0] + c * u[0] * sx / 2 - s * u[1] * sy / 2, w['c'][1] + s * u[0] * sx / 2 + c * u[1] * sy / 2)
            S.append((*P(a), *P(b)))
    S = np.array(S); d = S[:, 2:] - S[:, :2]; L = np.hypot(d[:, 0], d[:, 1]); keep = L > 1e-6
    S, d, L = S[keep], d[keep], L[keep]
    return S[:, :2], d / L[:, None], L          # a, unit direction, length

def nearest(X, A, U, L):
    """index of the nearest segment, its signed normal distance, for points X (n,2)."""
    rel = X[:, None, :] - A[None]                       # n,S,2
    s = np.clip(np.einsum('nsk,sk->ns', rel, U), 0, L[None])
    foot = A[None] + s[..., None] * U[None]
    dv = X[:, None, :] - foot
    dist = np.hypot(dv[..., 0], dv[..., 1]); j = np.argmin(dist, 1); r = np.arange(len(X))
    nrm = np.stack([-U[j, 1], U[j, 0]], 1)
    return j, np.einsum('nk,nk->n', dv[r, j], nrm), dist[r, j]

# ---------------------------------------------------------------- the registration
def place(poses, pts, off):
    out = []
    for (x, y, th), p in zip(poses, pts):
        c, s = math.cos(th + off), math.sin(th + off)
        out.append(np.stack([x + c * p[:, 0] - s * p[:, 1], y + s * p[:, 0] + c * p[:, 1]], 1))
    return out

def to_world(Q, phi, t):
    c, s = math.cos(phi), math.sin(phi); d = Q - t
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], 1)       # R(-phi)(Q - t): inverts internal = R(phi) world + t

GATE, SCALE = 0.25, 0.03     # metres: an inlier gate for the correspondences and the robust-loss scale (registration tooling, not a model term)
ITERS = 25
def register(Q, seg, phi0, t0, iters=None):
    iters = iters or ITERS
    A, U, L = seg
    phi, t = phi0, np.array(t0, float)
    for _ in range(iters):
        W0 = to_world(Q, phi, t); j, sd, dist = nearest(W0, A, U, L); m = dist < GATE
        if m.sum() < 50: return None
        nrm = np.stack([-U[j[m], 1], U[j[m], 0]], 1); Aj = A[j[m]]; Qm = Q[m]
        def res(p):
            Wp = to_world(Qm, p[0], p[1:]); return np.einsum('nk,nk->n', Wp - Aj, nrm)
        sol = least_squares(res, np.array([phi, *t]), loss='soft_l1', f_scale=SCALE, x_scale=[0.01, 0.1, 0.1])
        phi, t = float(sol.x[0]), sol.x[1:]
    d = nearest(to_world(Q, phi, t), A, U, L)[2]
    return phi, t, int(m.sum()), float(np.sqrt(np.mean(res(sol.x) ** 2))), int((d < 0.05).sum())   # last: points within 5 cm of a true wall

def register_ms(Q, seg, phi0, t0):
    """the fit from three starts (phi0, phi0 +- 3 deg); the one with the most points within 5 cm of a wall wins; also returns the starts' spread."""
    rs = [register(Q, seg, phi0 + d, t0) for d in (0.0, -0.05, 0.05)]; rs = [r for r in rs if r]
    best = max(rs, key=lambda r: r[4])
    return best, max(abs(r[0] - best[0]) for r in rs), sum(abs(r[0] - best[0]) > 0.1 / D for r in rs)

def block_boot(Qs, seg, phi, t, B, blk=10):
    n = len(Qs); out = []
    nb = max(n // blk, 1)
    for _ in range(B):
        idx = np.concatenate([np.arange(s, min(s + blk, n)) for s in rng.integers(0, max(n - blk, 1), nb)])
        r = register(np.concatenate([Qs[i] for i in idx]), seg, phi, t, iters=3)
        if r: out.append(r[0])
    return float(np.std(out, ddof=1)) if len(out) > 2 else math.nan

# ---------------------------------------------------------------- main
def main(argv):
    global ITERS
    B, only, out_csv, cache, tours = 24, None, None, os.path.expanduser('~/.cache/first_visit_nees'), yc1.TOURS
    i = 0
    while i < len(argv):
        if argv[i] == '--boot': B = int(argv[i + 1]); i += 1
        elif argv[i] == '--iters': ITERS = int(argv[i + 1]); i += 1
        elif argv[i] == '--only': only = argv[i + 1]; i += 1
        elif argv[i] == '--csv': out_csv = argv[i + 1]; i += 1
        elif argv[i] == '--cache': cache = argv[i + 1]; i += 1
        elif argv[i] == '--tours': tours = argv[i + 1]; i += 1
        i += 1
    # 1) YC1's rows (visits, staged edge, scorer frame), captured instead of printed; the recs give each visit's stay window
    recs_by = {}; rows = []
    orig = F.score_tour
    def spy(run, *a, **k):
        r = orig(run, *a, **k)
        for x in r: recs_by[(run.name[-3:], x.get('node'))] = x
        return r
    F.score_tour = spy; yc1.report = lambda rr, upd: rows.extend(rr)
    yc1.main(['--tours', tours, '--cache', cache]); F.score_tour = orig
    from aba_geometry import Frames, read_gauge, read_gt_error
    seg = wall_segments(F.DEFAULT_WBT)
    box = [r for r in rows if r['cls'] in ('nightstand', 'bed', 'toilet') and yc1.valid_wall(r)]
    print(f'visits with a true wall: {len(box)}; wall segments: {len(seg[0])}', file=sys.stderr)
    tapes, gts = {}, {}
    res = []
    OFF = None
    for r in sorted(box, key=lambda r: (r['tour'], r['t_out'])):
        if only and r['tour'] not in only.split(','): continue
        tour = r['tour']; run = R.run_path(f'{tours}/g1x_tour_{tour}')
        if tour not in tapes:
            tp = sorted(Path(run).glob('scan_tape_*.bin'))[0]; ts, pose, pts = read_tape(tp)
            ge = read_gt_error(run / 'gt_error.csv'); gtfull = {}
            import csv
            with open(run / 'gt_error.csv') as fh:
                for q in csv.DictReader(fh): gtfull[int(q['ts_ms'])] = (float(q['gt_x_world']), float(q['gt_y_world']), float(q['gt_theta']))
            tapes[tour] = (ts, pose, pts, ge, gtfull, read_gauge(run / 'proto_room.csv'))
        ts, pose, pts, ge, gtfull, gauge = tapes[tour]
        rec = recs_by[(tour, r['node'])]; t0, t1 = rec['t_in'], rec['t_out']
        fr = Frames(ge, gauge, t0, t1)
        sel = np.where((ts >= t0) & (ts <= t1))[0]; sel = sel[::3]       # every 3rd frame (~6 Hz): neighbouring scans are nearly the same view
        P = [pts[k][::12] for k in sel]; pe = pose[sel]
        if OFF is None:   # the lidar frame's quarter-turn relative to the pose heading: a discrete convention, chosen once on tour S01 by the exact pose
            best = None
            for off in (0.0, PI / 2, PI, -PI / 2):
                gp = [gtfull.get(int(ts[k])) for k in sel]; ok = [g is not None for g in gp]
                if not all(ok): continue
                Qg = np.concatenate(place([(g[0], g[1], g[2] - PI / 2) for g in gp], P, off))
                _, _, dist = nearest(Qg, *seg); sc = float(np.median(dist))
                if best is None or sc < best[0]: best = (sc, off)
            OFF = best[1]; print(f'lidar quarter-turn offset {OFF:.4f} rad (median wall distance {best[0]:.3f} m with the exact pose)', file=sys.stderr)
        Qs = place(pe, P, OFF)                                           # scans in the published (internal) frame, placed by the tape's estimated pose
        reg, spread, ndis = register_ms(np.concatenate(Qs), seg, fr.phi, fr.t)
        # null test: the same scans placed by the exact Webots pose; the rotation of that fit is the method's constant + noise
        gp = [gtfull.get(int(ts[k])) for k in sel]
        Qg = place([(g[0], g[1], g[2] - PI / 2) for g in gp], P, OFF)
        nul = register_ms(np.concatenate(Qg), seg, 0.0, [0.0, 0.0])[0]
        sb = block_boot(Qs, seg, reg[0], reg[1], B) if B else math.nan
        res.append(dict(r=r, fr=fr, phi_scan=reg[0], t_scan=reg[1], n_in=reg[2], rms=reg[3], spread=spread, ndis=ndis, null=nul[0] if nul else math.nan,
                        null_t=nul[1] if nul else None, boot=sb, nfr=len(sel)))
        print(f"{tour} {r['key']}: phi_scorer {fr.phi*D:+.3f} phi_scan {reg[0]*D:+.3f} null {math.degrees(nul[0]):+.3f} boot {math.degrees(sb) if sb==sb else 0:.3f} (n {len(sel)})", file=sys.stderr)
    report(res)
    if out_csv:
        with open(out_csv, 'w') as fh:
            fh.write('tour,key,room,phi_scorer,phi_scan,null,boot_sd,wall_err,th_staged\n')
            for q in res:
                r = q['r']; fh.write(f"{r['tour']},{r['key']},{r['room']},{q['fr'].phi:.6f},{q['phi_scan']:.6f},{q['null']:.6f},{q['boot']:.6f},{r['wall_err']:.6f},{r['th_staged']:.6f}\n")

# ---------------------------------------------------------------- tables
def report(res):
    nul = np.array([q['null'] for q in res]); b0 = float(np.mean(nul))
    print(f'\n## YC2 raw: {len(res)} graded visits with a true wall, registered from {sum(q["nfr"] for q in res)} tape frames (every 3rd, every 12th point)\n')
    print('### the registration itself: how well does a stay\'s scans fix rotation?\n')
    print(f'NULL TEST (scans placed by the exact Webots pose; the fitted rotation should be a constant, the lidar mount): mean b0 = {b0*D:+.3f} deg, '
          f'sd across the {len(res)} stays = {np.std(nul, ddof=1)*D:.3f} deg, range [{nul.min()*D:+.3f}, {nul.max()*D:+.3f}]')
    bs = np.array([q['boot'] for q in res]); sp = np.array([q['spread'] for q in res])
    print(f'block bootstrap over a stay\'s frames: median sd {np.nanmedian(bs)*D:.3f} deg (max {np.nanmax(bs)*D:.3f}); three starts (scorer phi, +-3 deg): the best-scoring fit is taken; '
          f'{int(sum(q["ndis"] > 0 for q in res))} of {len(res)} stays have a start that ends more than 0.1 deg away, largest spread {sp.max()*D:.2f} deg')
    print(f'median inliers {int(np.median([q["n_in"] for q in res]))} points, median rms of the fit {np.median([q["rms"] for q in res])*100:.1f} cm')
    # per-stay table
    seen = {}
    for q in res: seen.setdefault((q['r']['tour'], q['r']['room']), q)
    S = list(seen.values())
    print('\n### per (tour, room): the stay\'s rotation, scorer vs scans, and the staged edge in both frames (degrees; first graded visit of the stay)\n')
    print('| tour | room | phi_scorer | phi_scan - b0 | diff (scan - scorer) | boot sd | start spread | null (that stay) | wall_err scorer frame | wall_err scan frame |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    rows2 = []
    for q in sorted(S, key=lambda q: (q['r']['room'], q['r']['tour'])):
        r = q['r']; ph_s = q['fr'].phi; ph_i = q['phi_scan'] - b0
        # wall_err = fold(th_staged - (dir + phi + gth)); the independent frame replaces phi
        we_i = fold(r['wall_err'] + ph_s - ph_i, PI)
        rows2.append((r['room'], r['tour'], ph_s, ph_i, r['wall_err'], we_i, q['boot'], q['null'] - b0))
        print(f"| {r['tour']} | {r['room']} | {ph_s*D:+.3f} | {ph_i*D:+.3f} | {(ph_i-ph_s)*D:+.3f} | {q['boot']*D:.3f} | {q['spread']*D:.2f} | {(q['null']-b0)*D:+.3f} | {r['wall_err']*D:+.2f} | {we_i*D:+.2f} |")
    A = np.array([(x[2], x[3], x[4], x[5], x[6], x[7]) for x in rows2])
    dif = A[:, 1] - A[:, 0]
    w0, w1 = A[:, 2], A[:, 3]      # wall_err in the scorer frame, in the scan frame; w0 = w1 + dif
    tot = np.sum(w0 ** 2)
    print(f'\ndecomposition of the staged wall error, per stay w0 = w1 + dif (w1 = in the scan frame = the polygon\'s own part, dif = scan rotation - scorer rotation = the scorer\'s part): '
          f'sum w0^2 = {tot*D*D:.2f} deg^2 = sum w1^2 {np.sum(w1**2)/tot*100:.0f}% + 2 sum w1*dif {2*np.sum(w1*dif)/tot*100:+.0f}% + sum dif^2 {np.sum(dif**2)/tot*100:.0f}%')
    big = np.argsort(-np.abs(dif))[:2]; keep = np.setdiff1d(np.arange(len(A)), big)
    print(f'sensitivity: without the two stays with the largest |dif| ({[(rows2[i][1], rows2[i][0], round(dif[i]*D, 2)) for i in big]}): rms wall_err scorer {np.sqrt(np.mean(w0[keep]**2))*D:.2f}, '
          f'scan {np.sqrt(np.mean(w1[keep]**2))*D:.2f} deg; sum w1^2 / sum w0^2 = {np.sum(w1[keep]**2)/np.sum(w0[keep]**2):.2f}; bootstrap sd of the two: '
          f'{[round(rows2[i][6]*D, 2) for i in big]}')
    print(f'\nn stays {len(A)}; scorer rotation minus scan rotation: mean {-dif.mean()*D:+.3f} deg, sd {dif.std(ddof=1)*D:.3f}, rms {np.sqrt(np.mean(dif**2))*D:.3f}; '
          f'registration noise (null sd {np.std(A[:,5], ddof=1)*D:.3f}, median bootstrap {np.nanmedian(A[:,4])*D:.3f})')
    print(f'corr(phi_scorer, phi_scan) over stays: {np.corrcoef(A[:,0], A[:,1])[0,1]:+.3f}; regression slope phi_scan on phi_scorer: {np.polyfit(A[:,0], A[:,1], 1)[0]:+.2f}')
    print('\n### the staged wall direction error in the two frames, by room\n')
    print('| room | n | scorer frame: mean | sd | rms | scan frame: mean | sd | rms | corr(scorer, scan) | share of wall_err variance (about 0) that survives = sum(we_scan^2)/sum(we_scorer^2) |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for room in sorted({x[0] for x in rows2}) + ['ALL']:
        sub = A if room == 'ALL' else np.array([(x[2], x[3], x[4], x[5], x[6], x[7]) for x in rows2 if x[0] == room])
        a, b = sub[:, 2], sub[:, 3]
        print(f"| {room} | {len(a)} | {a.mean()*D:+.2f} | {a.std(ddof=1)*D:.2f} | {np.sqrt(np.mean(a**2))*D:.2f} | {b.mean()*D:+.2f} | {b.std(ddof=1)*D:.2f} | {np.sqrt(np.mean(b**2))*D:.2f} | "
              f"{np.corrcoef(a, b)[0,1]:+.2f} | {np.sum(b**2)/np.sum(a**2):.2f} |")
    # per visit: e_yaw in the independent frame
    print('\n### the yaw error and its wall part per visit, in the independent frame\n')
    print('| class | n | sum z^2 scorer frame | sum z^2 scan frame | wall term (sum (wall_err/sigma)^2) scorer | scan | mean NEES/DOF scorer | scan frame |')
    print('|---|---|---|---|---|---|---|---|')
    for cls in ('nightstand', 'bed', 'toilet', 'ALL'):
        qs = res if cls == 'ALL' else [q for q in res if q['r']['cls'] == cls]
        z0 = z1 = w0 = w1 = 0.0; n0 = []; n1 = []
        for q in qs:
            r = q['r']; dphi = (q['phi_scan'] - b0) - q['fr'].phi            # truth_pub yaw = world yaw + phi + gth  ->  the independent truth is shifted by dphi
            e1 = fold(r['e_yaw'] - dphi, 2 * PI); s = r['sig_yaw']; we1 = fold(r['wall_err'] - dphi, PI)
            z0 += (r['e_yaw'] / s) ** 2; z1 += (e1 / s) ** 2; w0 += (r['wall_err'] / s) ** 2; w1 += (we1 / s) ** 2
            n0.append(r['nees']); n1.append(r['nees_with'](e1))
        print(f"| {cls} | {len(qs)} | {z0:.1f} | {z1:.1f} | {w0:.1f} | {w1:.1f} | {np.mean(n0):.2f} | {np.mean(n1):.2f} |")
    # the pooled regression: is the scorer's wall error explained by the scorer-vs-scan rotation difference?
    we0 = A[:, 2]; we1 = A[:, 3]
    sl = np.sum(we0 * dif) / np.sum(dif * dif)
    print(f'\nregression of the scorer-frame wall_err on (phi_scan - phi_scorer) through the origin: slope {sl:+.2f} (wall_err = {sl:.2f} * dif would be -1 if the scorer rotation were all of it); '
          f'corr {np.corrcoef(we0, dif)[0,1]:+.3f}')

if __name__ == "__main__":
    main(sys.argv[1:])
