#!/usr/bin/env python3
"""YC1 (2026-11-27): signed-yaw check and parked share of Part A, offline, measurement only.
    python3 -I 2026-11-27-yaw-check.py [--tours DIR] [--replay BIN] [--cache DIR] [--csv OUT.csv]
Reuses the registered scorer (first_visit_nees.score_tour: same visits, same rows, same replay, same frame fit) and adds:
  exp 1  per graded first visit: signed e_yaw, sigma_yaw, z; the STAGED wall edge's direction (read from the replay dump's
         own frame record, published frame) against the TRUE wall's direction (the Webots Wall node, taken into the same
         published frame by the scorer's own Frames); the frame fit's yaw spread and a re-fit at the draw time.
  exp 2  per graded first visit: updates taken parked (gt_pose.tsv, bracketing samples identical) vs moving, per class.
Everything is read with float()/int()/struct (no locale)."""
import csv, math, os, struct, sys
import numpy as np
SCR = '/home/pbustos/robocomp/components/active_inference.worktrees/shared-engine/room_concept/estimation_scripts/scripts'
sys.path.insert(0, SCR)
import first_visit_nees as F
R, G = F.R, F.G
TOURS = '/home/pbustos/robocomp/components/active_inference/room_concept/estimation_scripts/crossing'
PI = math.pi

def wrap(a): return (a + PI) % (2 * PI) - PI
def fold(a, m):
    """a folded to (-m/2, m/2]  (m = pi: an undirected line; pi/2: a rectangle's axis)."""
    return (a + m / 2) % m - m / 2

# ---------------------------------------------------------------- replay dump: the staged wall of update k
def _rec(b):
    n = struct.unpack_from('<I', b, 0)[0]; o = 4; d = {}
    for _ in range(n):
        nl = struct.unpack_from('<H', b, o)[0]; o += 2; name = b[o:o + nl].decode(); o += nl
        L = struct.unpack_from('<Q', b, o)[0]; o += 8; d[name] = b[o:o + L]; o += L
    return d

def staged_walls(dump):
    """[dict(ok, p, n, length, edge, room)] one per UpdateEv (event type 4 = variant index 3 + 1), in dump order."""
    b = open(dump, 'rb').read()
    assert b[:7] == b'RCRPLAY'
    o, out = 12, []
    while o < len(b):
        t, L = struct.unpack_from('<IQ', b, o); o += 12
        if t == 4:
            w = _rec(_rec(_rec(b[o:o + L])['frame'])['wall'])
            out.append(dict(ok=bool(w['ok'][0]), p=struct.unpack('<2f', w['p']), n=struct.unpack('<2f', w['n']),
                            length=struct.unpack('<f', w['length_m'])[0], edge=struct.unpack('<i', w['edge'])[0],
                            room=struct.unpack('<Q', w['room'])[0]))
        o += L
    return out

# ---------------------------------------------------------------- the true walls (Webots Wall nodes)
def true_walls(wbt):
    import wbt_rooms as W
    if os.path.abspath(wbt) != os.path.abspath(W.WBT): W.WBT = wbt
    walls, _, _ = W.parse_wbt(wbt)
    out = []
    for w in walls:
        sx, sy = w['size']                           # thickness x length (wbt_rooms docstring); the long axis is the wall's line
        a = w['ang'] if sx >= sy else w['ang'] + PI / 2
        out.append(dict(name=w['name'], c=np.array(w['c']), dir=a, half_len=max(sx, sy) / 2, half_th=min(sx, sy) / 2))
    return out

def nearest_true_wall(walls, pt):
    """(wall, distance from pt to the wall's centre segment): the Webots Wall whose centre line is nearest to pt (world)."""
    best = None
    for w in walls:
        u = np.array([math.cos(w['dir']), math.sin(w['dir'])]); d = pt - w['c']
        s = float(np.clip(d @ u, -w['half_len'], w['half_len'])); q = w['c'] + s * u
        dist = float(np.linalg.norm(pt - q))
        if best is None or dist < best[1]: best = (w, dist)
    return best

# ---------------------------------------------------------------- robot speed
def gt_speed_index(run):
    s = F.read_gt_pose(run)                               # (t, x, y)
    yaw = {}
    for r in R.read_tsv(run / 'gt_pose.tsv'):
        try: yaw[int(r['t_ms'])] = float(r['yaw'])
        except (KeyError, ValueError, TypeError): pass
    return [(t, x, y, yaw.get(t, 0.0)) for t, x, y in s]

def motion_at(idx, ts, t):
    """(dxy, dyaw, dt): displacement of the robot between the two gt_pose samples that bracket t (ms)."""
    import bisect
    i = min(max(bisect.bisect_right(ts, t) - 1, 0), len(ts) - 2)
    a, b = idx[i], idx[i + 1]
    return math.hypot(b[1] - a[1], b[2] - a[2]), abs(wrap(b[3] - a[3])), (b[0] - a[0]) / 1000.0

# ---------------------------------------------------------------- the replay rows: the cache of the scoring binary
def cached_runner(dump, replay_bin, cache):
    """Same contract as first_visit_nees.replay_rows, but WITHOUT running a binary. The replay tool in this worktree was rebuilt
    on 2026-11-27 21:14 (CW1's containment changed the engine), so it is no longer bit-identical to the live batch and the scorer's
    own gate rejects every visit. The cache of the binary that scored g1x_tour_nees.txt (built 2026-11-26 19:54, HEAD 9dfbf82d,
    --theta-cov) is still on disk, one csv/rc pair per dump; this picks, for a dump, the cached run with exit 0 whose every row has
    dth_live = dP_live = 0 (the scorer's own bit-identity gate) and the S_i_j covariance columns. None qualifying -> rc 1."""
    from pathlib import Path
    stem = Path(dump).stem; best = None
    for c in sorted(Path(cache).glob(f'{stem}_*.csv')):
        rcp = c.with_suffix('.rc')
        if not rcp.exists() or int(rcp.read_text()) != 0: continue
        with open(c) as fh: rows = list(csv.DictReader(fh))
        if not rows or 'S_0_0' not in rows[0]: continue
        try:
            if any(float(r['dth_live']) != 0.0 or float(r['dP_live']) != 0.0 for r in rows): continue
        except (TypeError, ValueError): continue   # a truncated csv
        if best is None or c.stat().st_mtime < best[0]: best = (c.stat().st_mtime, rows, c.name)   # the oldest = the batch's scorer
    if best is None: return [], 1, 'no cached bit-identical replay'
    return best[1], 0, ''

# ---------------------------------------------------------------- main
def main(argv):
    replay, cache, out_csv, tours = F.DEFAULT_REPLAY, os.path.expanduser('~/.cache/first_visit_nees'), None, TOURS
    i = 0
    while i < len(argv):
        if argv[i] == '--tours': tours = argv[i + 1]; i += 1
        elif argv[i] == '--replay': replay = argv[i + 1]; i += 1
        elif argv[i] == '--cache': cache = argv[i + 1]; i += 1
        elif argv[i] == '--csv': out_csv = argv[i + 1]; i += 1
        i += 1
    wbt = F.DEFAULT_WBT
    walls_true = true_walls(wbt)
    rows, upd = [], []
    for k in range(1, 13):
        run = R.run_path(f'{tours}/g1x_tour_S{k:02d}')
        recs = F.score_tour(run, wbt, replay, cache, runner=cached_runner)
        gt = gt_speed_index(run); gts = [g[0] for g in gt]
        gt_err = R.read_gt_error if False else None
        from aba_geometry import read_gt_error, read_gauge
        ge = read_gt_error(run / 'gt_error.csv')
        for rec in recs:
            if rec['status'] != 'ok': continue
            cls, node = rec['cls'], rec['node']
            cr, legacy = F.class_rows(run, cls)
            ups = R.updates(cr, node); inside = [u for u in ups if u['t'] <= rec['t_out']]
            # ---- exp 2: parked vs moving at every update of the visit
            runlen, prev_parked = 0, False
            for j, u in enumerate(inside):
                dxy, dyaw, dt = motion_at(gt, gts, u['t'])
                lo = int(u.get('lidar_only', 0)) if not legacy else 0
                upd.append(dict(tour=run.name[-3:], cls=cls, key=rec['key'], j=j, n=len(inside), dxy=dxy, dyaw=dyaw, dt=dt,
                                lidar_only=lo, legacy=int(legacy)))
            if rec.get('label'): continue                         # the legacy diagonal table: no full P, no dump
            # ---- exp 1
            dump = F.dump_for(run, node)[0]
            rr, rc, _ = cached_runner(dump, replay, cache)
            kd = len(inside) - 1
            P = F.cov_from_row(rr[kd]); est = [float(rr[kd][c]) for c in F.DOFS]; sig = [math.sqrt(P[j, j]) for j in range(6)]
            T = G.object_truth(wbt, rec['key'])
            m = R.match_node_to_truth(run, cr, {'A1': {'t_claim_ms': rec['t_in'], 't_dwell1_ms': rec['t_out']},
                                                 'A2': {'t_dwell1_ms': rec['t_out']}}, T)
            fr = m['frames']
            truth_pub = fr.world_to_pub(T['cx'], T['cy'], T['yaw_theta_world'], rec['t_out'])
            e, Pg = F.graded_error_full(est, sig, P, T, truth_pub)
            q_full = F.nees_full(e, Pg)[0]
            e2 = list(e); e2[5] = 0.0
            q_noyaw = F.nees_full(e2, Pg)[0]
            r_nees = (lambda Pg_, e_: (lambda yv: F.nees_full([*e_[:5], yv], Pg_)[0]))(Pg, list(e))
            sw = staged_walls(dump)[kd]
            r_ = dict(tour=run.name[-3:], cls=cls, key=rec['key'], room=rec['room'], node=node, nees=rec['nees'], nees_chk=q_full,
                      e_yaw=float(e[5]), sig_yaw=math.sqrt(Pg[5, 5]), p_yaw=Pg[5, 5], q_full=q_full, q_noyaw=q_noyaw,
                      est_yaw=est[5], truth_yaw=truth_pub[2], t_out=rec['t_out'], t_draw=inside[kd]['t'],
                      phi=fr.phi, fit_rms=fr.rms, fit_n=fr.n, gth=fr.gauge(rec['t_out'])[2])
            # frame fit's own yaw spread: the per-sample heading offset d_i of the registered fit, over the stay
            win = [g for g in ge if rec['t_in'] <= g[0] <= rec['t_out']]
            win_reg = [g for g in ge if rec['t_in'] <= g[0] <= rec['t_out']]   # the registered window (A1 claim .. A2 dwell end)
            d = np.array([wrap(g[3] - (g[6] - PI / 2)) for g in win_reg]) if win_reg else np.array([fr.phi])
            dd = np.array([wrap(x - fr.phi) for x in d])
            r_.update(d_sd=float(dd.std()), d_n=len(dd), d_mean_off=float(dd.mean()),
                      d_drift=float((dd[-max(len(dd) // 3, 1):].mean() - dd[:max(len(dd) // 3, 1)].mean())) if len(dd) > 3 else math.nan)
            # an independent rotation for the same window: Kabsch on the POSITIONS (est vs gt), not on the heading column
            Wp = np.array([[g[4], g[5]] for g in win_reg]); Ep = np.array([[g[1], g[2]] for g in win_reg])
            if len(Wp) > 3:
                Wc, Ec = Wp - Wp.mean(0), Ep - Ep.mean(0)
                r_['phi_pos'] = math.atan2(float(np.sum(Wc[:, 0] * Ec[:, 1] - Wc[:, 1] * Ec[:, 0])), float(np.sum(Wc * Ec)))
                r_['extent'] = float(np.sqrt(np.mean(np.sum(Wc ** 2, 1))))
            else: r_['phi_pos'], r_['extent'] = math.nan, 0.0
            # the heading offset of the whole run, of the stay's first and last 5 s (references the polygon might have been built at)
            def circ(sel):
                v = [wrap(g[3] - (g[6] - PI / 2)) for g in sel]
                return math.atan2(sum(map(math.sin, v)), sum(map(math.cos, v))) if v else math.nan
            Wr = np.array([[g[4], g[5]] for g in ge]); Er = np.array([[g[1], g[2]] for g in ge]); Wc, Ec = Wr - Wr.mean(0), Er - Er.mean(0)
            r_['phi_pos_run'] = math.atan2(float(np.sum(Wc[:, 0] * Ec[:, 1] - Wc[:, 1] * Ec[:, 0])), float(np.sum(Wc * Ec)))   # Kabsch over the WHOLE run
            r_['extent_run'] = float(np.sqrt(np.mean(np.sum(Wc ** 2, 1))))
            r_['phi_run'] = circ(ge); r_['phi_first5'] = circ([g for g in win_reg if g[0] <= rec['t_in'] + 5000])
            r_['phi_last5'] = circ([g for g in win_reg if g[0] >= rec['t_out'] - 5000]); r_['phi_before'] = circ([g for g in ge if rec['t_in'] - 20000 <= g[0] < rec['t_in']])
            # re-fit at the draw: the heading offset in +-3 s about the draw
            td = inside[kd]['t']
            loc = [wrap(g[3] - (g[6] - PI / 2)) for g in ge if abs(g[0] - td) <= 3000]
            r_['phi_loc'] = (math.atan2(sum(map(math.sin, loc)), sum(map(math.cos, loc))) if loc else fr.phi)
            # staged wall vs true wall in the published frame
            if sw['ok']:
                p = np.array(sw['p'], float); n = np.array(sw['n'], float)
                th_staged = math.atan2(n[0], -n[1])                      # direction of the edge (the normal rotated by +90 deg)
                wx, wy, _ = fr.pub_to_world(p[0], p[1], 0.0, td)
                w, dist = nearest_true_wall(walls_true, np.array([wx, wy]))
                th_true = w['dir'] + fr.phi + r_['gth']
                r_.update(wall_ok=1, wall_len=sw['length'], wall_edge=sw['edge'], wall_dist=dist, wall_name=w['name'],
                          wall_err=fold(th_staged - th_true, PI), th_staged=th_staged, th_true=th_true)
                # the object's own truth relative to that wall: back direction vs the wall normal (0 = flush)
                bk = math.atan2(T['back'][1], T['back'][0]); wn = th_true + PI / 2
                r_['obj_vs_wall_true'] = fold(bk - wn, PI / 2)
            else:
                r_.update(wall_ok=0)
            r_['nees_with'] = r_nees
            rows.append(r_)
    if out_csv:
        with open(out_csv, 'w') as fh:
            keys = sorted({k for r in rows for k in r} - {'nees_with'}); w = csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore'); w.writeheader(); w.writerows(rows)
    report(rows, upd)

# ---------------------------------------------------------------- tables
D = 180 / PI
WALL_TOL_M = 0.3   # a staged edge counts as having a TRUE Webots Wall node only if one lies within this distance of it (room_1's walls are meshes: nearest box is ~3 m away)
def valid_wall(r): return bool(r.get('wall_ok')) and r['wall_dist'] <= WALL_TOL_M
def mean(x): return float(np.mean(x)) if len(x) else math.nan
def sd(x): return float(np.std(x, ddof=1)) if len(x) > 1 else math.nan

def report(rows, upd):
    box = [r for r in rows if r['cls'] in ('nightstand', 'bed', 'toilet', 'refrigerator', 'hood')]
    print(f'## Exp 1 raw: {len(rows)} graded full-P visits ({len(box)} of the four-sided box classes; chair {sum(r["cls"]=="chair" for r in rows)})')
    print('\n### per visit (degrees)\n')
    print('| tour | class | key | room | e_yaw | sigma_yaw | z | wall_err | e-wall | edge | len m | wall d m | obj-vs-wall truth | phi_loc-phi | d_sd | NEES/DOF | w/o yaw |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for r in sorted(rows, key=lambda r: (r['cls'], r['tour'], r['key'])):
        we = r.get('wall_err', math.nan)
        print(f"| {r['tour']} | {r['cls']} | {r['key']} | {r['room']} | {r['e_yaw']*D:+.2f} | {r['sig_yaw']*D:.2f} | {r['e_yaw']/r['sig_yaw']:+.2f} | "
              f"{we*D:+.2f} | {(r['e_yaw']-we)*D:+.2f} | {r.get('wall_edge','-')} | {r.get('wall_len',math.nan):.1f} | {r.get('wall_dist',math.nan):.2f} | "
              f"{r.get('obj_vs_wall_true',math.nan)*D:+.1f} | {wrap(r['phi_loc']-r['phi'])*D:+.2f} | {r['d_sd']*D:.2f} | {r['nees']:.2f} | {r['q_noyaw']:.2f} |")

    print('\n### per class: yaw error and what subtracting each candidate does to the yaw term\n')
    print('| class | n | mean e_yaw deg | sd deg | mean sigma_yaw deg | mean z^2 (yaw term) | after minus wall_err | after minus tour/room mean | corr(e, wall_err) | slope | mean NEES/DOF | w/o yaw | w/o yaw share |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for cls in ('nightstand', 'bed', 'toilet', 'refrigerator', 'hood', 'chair'):
        rs = [r for r in rows if r['cls'] == cls and valid_wall(r)]
        if not rs: continue
        e = np.array([r['e_yaw'] for r in rs]); s = np.array([r['sig_yaw'] for r in rs]); we = np.array([r['wall_err'] for r in rs])
        z2 = float(np.mean((e / s) ** 2)); z2w = float(np.mean(((e - we) / s) ** 2))
        grp = {}
        for r in rs: grp.setdefault((r['tour'], r['room']), []).append(r)
        # leave-one-out group mean (so a singleton group removes nothing and a pair does not remove itself)
        sub = []
        for r in rs:
            o = [x['e_yaw'] for x in grp[(r['tour'], r['room'])] if x is not r]
            sub.append((r['e_yaw'] - (np.mean(o) if o else 0.0)) / r['sig_yaw'])
        z2g = float(np.mean(np.square(sub)))
        cc = float(np.corrcoef(e, we)[0, 1]) if len(rs) > 2 and we.std() > 0 and e.std() > 0 else math.nan
        sl = float(np.polyfit(we, e, 1)[0]) if len(rs) > 2 and we.std() > 0 else math.nan
        q = mean([r['nees'] for r in rs]); qn = mean([r['q_noyaw'] for r in rs])
        print(f"| {cls} | {len(rs)} | {e.mean()*D:+.2f} | {e.std(ddof=1)*D:.2f} | {s.mean()*D:.2f} | {z2:.2f} | {z2w:.2f} | {z2g:.2f} | {cc:+.2f} | {sl:+.2f} | {q:.2f} | {qn:.2f} | {(q-qn)/q*100:.0f}% |")

    report2(rows)
    report_exp2(upd)

def report2(rows):
    box = [r for r in rows if r['cls'] in ('nightstand', 'bed', 'toilet', 'refrigerator') and valid_wall(r)]
    skipped = [r for r in rows if r['cls'] in ('nightstand', 'bed', 'toilet', 'refrigerator', 'hood') and not valid_wall(r)]
    print(f'\nvisits with a Webots Wall node within {WALL_TOL_M} m of the staged edge: {len(box)}; without (no true wall direction): '
          f'{len(skipped)} ({sorted({(r["cls"], r["room"]) for r in skipped})})')
    # ---- the wall: one staged edge per (tour, room); what it is against the true wall, and what the frame fit says
    print('\n### per (tour, room): the staged wall edge against the true wall, the objects on it, the frame fit (degrees)\n')
    print('| tour | room | edge | L m | wall_err | e_yaw of the objects | phi_pos - phi_head | extent m | d_sd (head offset sd in the stay) |')
    print('|---|---|---|---|---|---|---|---|---|')
    grp = {}
    for r in box: grp.setdefault((r['tour'], r['room'], r['wall_edge']), []).append(r)
    W = []
    for (t, room, ed), rs in sorted(grp.items()):
        r0 = rs[0]; dp = wrap(r0['phi_pos'] - r0['phi']) * D if r0['phi_pos'] == r0['phi_pos'] else math.nan
        W.append(dict(tour=t, room=room, we=r0['wall_err'] * D, dp=dp, ext=r0['extent'], dsd=r0['d_sd'] * D, es=[x['e_yaw'] * D for x in rs]))
        print(f"| {t} | {room} | {ed} | {r0['wall_len']:.1f} | {r0['wall_err']*D:+.2f} | " + ', '.join(f"{x['cls'][:3]} {x['e_yaw']*D:+.2f}" for x in rs)
              + f" | {dp:+.2f} | {r0['extent']:.2f} | {r0['d_sd']*D:.2f} |")
    print('\n### the staged wall error by room, across tours (is it a property of the room, or of the tour?)\n')
    print('| room | edge(s) | tours | mean wall_err | sd | rms | negative / positive | rms of (phi_pos - phi_head) | mean d_sd |')
    print('|---|---|---|---|---|---|---|---|---|')
    for room in sorted({w['room'] for w in W}):
        ws = [w for w in W if w['room'] == room]; a = np.array([w['we'] for w in ws]); dp = np.array([w['dp'] for w in ws])
        print(f"| {room} | {sorted({r['wall_edge'] for r in box if r['room'] == room})} | {len(ws)} | {a.mean():+.2f} | {a.std(ddof=1):.2f} | {np.sqrt(np.mean(a**2)):.2f} | "
              f"{int((a<0).sum())} / {int((a>0).sum())} | {np.sqrt(np.nanmean(dp**2)):.2f} | {np.mean([w['dsd'] for w in ws]):.2f} |")
    a = np.array([w['we'] for w in W]); print(f"\nall rooms: n {len(a)}, mean {a.mean():+.2f}, rms {np.sqrt(np.mean(a**2)):.2f} deg;"
        f" an ANOVA on room: between-room share of the variance about the grand mean = "
        f"{sum(len([w for w in W if w['room']==rm])*(np.mean([w['we'] for w in W if w['room']==rm])-a.mean())**2 for rm in {w['room'] for w in W})/np.sum((a-a.mean())**2):.2f}")
    # same tour, different rooms: is there a tour-level (frame) component?
    bt = {}
    for w in W: bt.setdefault(w['tour'], []).append(w['we'])
    pr = [(x[0], x[1]) for x in bt.values() if len(x) >= 2]
    print(f"within-tour pairs of rooms (a tour-level frame error would make them agree): n {len(pr)}, corr {np.corrcoef(np.array(pr).T)[0,1]:+.2f}")
    # a second frame rotation, from the whole run's positions (extent of metres, not decimetres), and the heading mean of the whole run
    print('\n### the staged wall error under other frame rotations (degrees; per (tour, room) first row)\n')
    print('| frame rotation used for the true wall | rms of wall_err | mean | sd | corr with the registered wall_err |')
    print('|---|---|---|---|---|')
    seen = {}
    for r in box: seen.setdefault((r['tour'], r['room']), r)
    S = list(seen.values()); base = np.array([r['wall_err'] for r in S])
    for name, key in (('registered: heading offset averaged over the stay', 'phi'), ('heading offset averaged over the whole run', 'phi_run'),
                      ('Kabsch on positions over the stay (extent ~0.7-2 m)', 'phi_pos'), ('Kabsch on positions over the whole run (extent ~%.0f m)' % np.median([r['extent_run'] for r in S]), 'phi_pos_run')):
        a = np.array([fold(r['wall_err'] + r['phi'] - r[key], PI) for r in S])
        print(f"| {name} | {np.sqrt(np.mean(a**2))*D:.2f} | {a.mean()*D:+.2f} | {a.std(ddof=1)*D:.2f} | {np.corrcoef(a, base)[0,1]:+.2f} |")
    # what the engine's own sigma says about the same walls
    lk = [r for r in box]
    print('\nengine sigma_yaw of a wall-locked object = 0.02 m / L: ' + ', '.join(f"{rm} L {np.mean([r['wall_len'] for r in box if r['room']==rm]):.1f} m -> {math.degrees(0.02/np.mean([r['wall_len'] for r in box if r['room']==rm])):.2f} deg, "
          f"rms wall_err {np.sqrt(np.mean([r['wall_err']**2 for r in box if r['room']==rm]))*D:.2f} deg" for rm in sorted({r['room'] for r in box})))
    print(f"frame fit: median fit_rms {np.median([r['fit_rms'] for r in box])*100:.1f} cm, median extent {np.median([r['extent'] for r in box]):.2f} m, median d_sd {np.median([r['d_sd'] for r in box])*D:.2f} deg")
    # ---- the decomposition
    print('\n### the yaw term: e_yaw = wall_err + e_rel   (e_rel = the object against the staged wall, frame-free)\n')
    print('| class | n | sum z^2 | sum (wall_err/sigma)^2 | sum (e_rel/sigma)^2 | share of sum z^2 carried by the wall term (1 - rel/total) | median e_rel deg | n with |e_rel| < 0.3 deg |')
    print('|---|---|---|---|---|---|---|---|')
    tot = dict(z=0, w=0, r=0, n=0)
    for cls in ('nightstand', 'bed', 'toilet', 'ALL'):
        rs = box if cls == 'ALL' else [r for r in box if r['cls'] == cls]
        e = np.array([r['e_yaw'] for r in rs]); s = np.array([r['sig_yaw'] for r in rs]); we = np.array([r['wall_err'] for r in rs]); er = e - we
        z = np.sum((e / s) ** 2); zw = np.sum((we / s) ** 2); zr = np.sum((er / s) ** 2)
        print(f"| {cls} | {len(rs)} | {z:.1f} | {zw:.1f} | {zr:.1f} | {(1 - zr / z) * 100:.0f}% | {np.median(er)*D:+.2f} | {int((np.abs(er) < 0.3 / D).sum())} |")
    # ---- common mode within (tour, room)
    print('\n### are the objects of one wall in one tour moving together?\n')
    pairs = [(a['e_yaw'] * D, b['e_yaw'] * D, a['cls'], b['cls']) for g in grp.values() for i, a in enumerate(g) for b in g[i + 1:]]
    pa = np.array([(x[0], x[1]) for x in pairs])
    print(f"pairs of graded objects on the same staged edge in the same tour: n {len(pairs)}; corr of their e_yaw {np.corrcoef(pa.T)[0,1]:+.3f}; "
          f"same sign {int((np.sign(pa[:,0])==np.sign(pa[:,1])).sum())}/{len(pairs)}; mean |difference| {np.mean(np.abs(pa[:,0]-pa[:,1])):.2f} deg "
          f"against mean |e_yaw| {np.mean(np.abs(pa)):.2f} deg")
    e_all = np.array([r['e_yaw'] for r in box]) * D
    gm = {k: np.mean([x['e_yaw'] for x in v]) * D for k, v in grp.items() if len(v) >= 2}
    num = sum(sum((x['e_yaw'] * D - gm[k]) ** 2 for x in v) for k, v in grp.items() if len(v) >= 2)
    den = sum(sum((x['e_yaw'] * D) ** 2 for x in v) for k, v in grp.items() if len(v) >= 2)
    print(f"over the {sum(len(v) for k, v in grp.items() if len(v) >= 2)} objects in groups of >= 2: sum of squared e_yaw {den:.1f}; about their group means {num:.1f} "
          f"=> the group mean carries {(1 - num / den) * 100:.0f}% of the squared yaw error")
    # the same for the nightstand+bed rows only, and e_yaw explained by wall_err by regression through the origin
    we = np.array([r['wall_err'] for r in box]) * D; ee = e_all
    print(f"regression of e_yaw on wall_err through the origin over all {len(box)}: slope {np.sum(we*ee)/np.sum(we*we):.2f}; "
          f"R^2 about zero {1-np.sum((ee-we)**2)/np.sum(ee**2):.2f} (with slope fixed at 1)")
    # ---- the registered full-P NEES with the wall term removed, and with the frame-fit candidates removed
    print('\n### full-P NEES/DOF if the yaw error is replaced (per class, mean over the graded visits)\n')
    print('| class | n | registered | e_yaw -> e_rel (wall_err removed) | yaw DOF removed (6 -> 5) | e_yaw -> e_yaw - (phi_pos - phi_head) | e_yaw -> e_yaw - (phi_loc - phi_head) |')
    print('|---|---|---|---|---|---|---|')
    for cls in ('nightstand', 'bed', 'toilet'):
        rs = [r for r in box if r['cls'] == cls]; out = []
        for fn in (lambda r: r['e_yaw'], lambda r: r['e_yaw'] - r['wall_err'], lambda r: 0.0,
                   lambda r: r['e_yaw'] - (wrap(r['phi_pos'] - r['phi']) if r['phi_pos'] == r['phi_pos'] else 0.0),
                   lambda r: r['e_yaw'] - wrap(r['phi_loc'] - r['phi'])):
            out.append(np.mean([r['nees_with'](fn(r)) for r in rs]))
        print(f"| {cls} | {len(rs)} | {out[0]:.2f} | {out[1]:.2f} | {out[2]:.2f} | {out[3]:.2f} | {out[4]:.2f} |")

def report_exp2(upd):
    print('\n## Exp 2: parked share of Part A (graded first-visit windows)\n')
    print('Parked at an update = the two gt_pose.tsv samples (about 1.16 s apart) that bracket the update are identical to the file\'s '
          'four-decimal rounding (|dxy| <= 1e-3 m and |dyaw| <= 1e-3 rad). Translation-only drops the yaw condition. The speed columns are '
          'dxy / dt of the same bracket.\n')
    print('| class | visits | updates | parked (xy and yaw) | translation-still | speed < 0.005 m/s | < 0.02 | < 0.05 | LiDAR-only | parked and LiDAR-only | repeats (parked updates after the first of a parked run) | visits whose DRAW is parked |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|')
    allr = {}
    for u in upd: allr.setdefault(u['cls'], []).append(u)
    def line(cls, us):
        n = len(us)
        pk = [u['dxy'] <= 1e-3 and u['dyaw'] <= 1e-3 for u in us]
        ts = [u['dxy'] <= 1e-3 for u in us]
        sp = lambda t: sum(u['dxy'] / max(u['dt'], 1e-3) < t for u in us)
        rep, prev_key = 0, None
        draw = 0; vis = {}
        for u, p in zip(us, pk):
            k = (u['tour'], u['key'])
            if k != prev_key: prev_key, prev = k, False
            if p and prev: rep += 1
            prev = p
            vis.setdefault(k, []).append((u['j'], p))
        draw = sum(max(v)[1] for v in vis.values())
        lo = sum(u['lidar_only'] for u in us); plo = sum(p and u['lidar_only'] for u, p in zip(us, pk))
        pc = lambda x: f"{x} ({x/n*100:.0f}%)"
        return f"| {cls} | {len(vis)} | {n} | {pc(sum(pk))} | {pc(sum(ts))} | {pc(sp(0.005))} | {pc(sp(0.02))} | {pc(sp(0.05))} | {pc(lo)} | {pc(plo)} | {pc(rep)} | {draw}/{len(vis)} |"
    for cls in ('nightstand', 'bed', 'toilet', 'refrigerator', 'hood', 'chair', 'table'):
        if cls in allr: print(line(cls + (' (legacy log)' if allr[cls][0]['legacy'] else ''), allr[cls]))
    print(line('all', [u for u in upd]))
    print('\nmoving share by update index within the visit (first third / middle / last third), parked fraction:')
    for cls in ('nightstand', 'bed', 'toilet', 'refrigerator', 'chair'):
        us = allr.get(cls, [])
        th = [[], [], []]
        for u in us: th[min(int(3 * u['j'] / max(u['n'], 1)), 2)].append(u['dxy'] <= 1e-3 and u['dyaw'] <= 1e-3)
        print(f"  {cls}: " + ' / '.join(f"{np.mean(t)*100:.0f}% (n {len(t)})" for t in th))

if __name__ == "__main__":
    main(sys.argv[1:])
