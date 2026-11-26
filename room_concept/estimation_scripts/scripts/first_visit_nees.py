#!/usr/bin/env python3
"""AuRo test (a), shared-engine part 1 grading: first-visit 6-DOF full-covariance NEES over tours.
    python3 -I first_visit_nees.py <tour_run_dir>... [--wbt W] [--replay BIN] [--cache DIR] [--smoke] [--classes a,b]
Definitions pre-registered in .superpowers/sdd/2026-11-23-shared-engine/task-G1a-scorer-brief.md (not to be changed after a result is seen):

  statistic   per class, the mean over first visits of NEES/DOF = e^T P^-1 e / 6, e = graded quantity - mesh truth (6 DOF: cx cy H w h yaw),
              P = the full 6x6 theta block of the node's posterior. Pass if the mean is in [0.5, 2] and n >= 10 graded visits.
  instances   every instance of the seven classes in the world file (mesh_extents.world_nodes); truth = aba_geometry.object_truth, the
              code that writes aba_plan_<DEF>.json (bed h an interval, chair yaw in the countable convention). The error is
              aba_runlog.graded_errors (bed interval, c2v minimum over a table's group images). For a c2v class P is permuted
              with the image (perm) so e and P stay in the same description.
  node match  aba_runlog.match_node_to_truth itself (chi2_2 99 % on (cx, cy) at the visit's end). Unmatched = reported, counts as a failure.
  first visit the robot's FIRST STAY in the room that contains the object: gt_pose.tsv (the Webots pose, one row per ~1 s) against the
              world's room polygons (the same polygons object_truth uses). The stay runs from the first sample inside the room
              to the LAST sample inside it before the first sample in a different room (samples in no polygon, i.e. a doorway,
              are neutral). The matched node's visit = its updates from its first update to the end of that stay. One draw per
              (tour, instance): the posterior at the node's LAST update of the visit. Rows inside a visit are not independent.
  frames      theta is in the published room frame; the truth is taken INTO that frame (Frames.world_to_pub, fitted on gt_error over the
              stay, gauge at the stay's end), so e and P are in the same frame and no rotation of P is needed.
  full P      NOT in the dump (it stores diag P only): recovered by driving the replay tool with --theta-cov (the engine's own
              covariance(), bit-identical on the build that wrote the dump). The draw is the k-th update of the node in the ai2 log
              (aba_runlog.updates) = the k-th update of its dump; the cycle, gated and lidar_only of the two must agree.
              Valid only if the tool exits 0 and the draw's dth_live and dP_live are exactly 0 (bit-identical to the live run), else
              the visit is "no full P" (--smoke waives that, for runs whose dumps predate the engine in the tool).
  table       the legacy binary writes no dumps: its NEES is DIAGONAL from the ai2 log sigma and labelled "old engine, diagonal".
              Any other class without a usable dump: the visit is "no full P", never silently diagonal.
  output      per class: visits, matched, graded, mean NEES/DOF, bootstrap 95 % CI over visits (percentile, 10000 resamples of the
              graded visits, fixed seed: a computation, not a bound), per-DOF mean e^2/P_kk, PASS/FAIL. No other statistic, no
              outlier removal. PASS needs: n graded >= 10, no unmatched visit, no 'no full P' visit, mean in [0.5, 2].
Every number is read with python float()/int() or struct-free text parsing, which does not read the locale.
"""
import csv, hashlib, math, os, subprocess, sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'tools', 'gt_mesh'))
import aba_runlog as R
import aba_geometry as G

CLASSES = ('nightstand', 'bed', 'toilet', 'chair', 'table', 'refrigerator', 'hood')
DOFS = ('cx', 'cy', 'H', 'w', 'h', 'yaw')
DEFAULT_WBT = os.environ.get('WBT_PATH', '/home/pbustos/robocomp/components/webots-shadow/worlds/piso/threerooms-piso.wbt')
DEFAULT_REPLAY = os.path.normpath(os.path.join(HERE, '..', '..', '..', 'countable_concept', 'tests', 'build', 'replay'))
PASS_LO, PASS_HI, N_MIN = 0.5, 2.0, 10
BOOT_N, BOOT_SEED = 10000, 0
COV_COLS = [f'S_{i}_{j}' for i in range(6) for j in range(i, 6)]


# ------------------------------------------------------------------------------------------------ the statistic
def nees_full(e, P):
    """(NEES/DOF, per-DOF e_k^2 / P_kk) of error e (6) under the full covariance P (6x6). P must be positive definite."""
    e = np.asarray(e, float); P = np.asarray(P, float)
    L = np.linalg.cholesky(P)                       # raises LinAlgError when P is not positive definite
    z = np.linalg.solve(L, e)
    return float(z @ z) / len(e), (e ** 2 / np.diag(P))


def nees_diag(e, sig):
    e = np.asarray(e, float); s = np.asarray(sig, float)
    return float(np.sum((e / s) ** 2)) / len(e), (e / s) ** 2


def cov_from_row(row):
    """6x6 symmetric from the replay's upper-triangle columns S_i_j."""
    P = np.zeros((6, 6))
    for i in range(6):
        for j in range(i, 6): P[i, j] = P[j, i] = float(row[f'S_{i}_{j}'])
    return P


def graded_error_full(est, sig, P, T, truth_pub):
    """(e, P in the TRUTH's DOF labels). The error is aba_runlog.graded_errors; for a c2v class the covariance follows the image
    graded_errors picked: perm[k] = the estimate's DOF that stands for DOF k, so P_img = P[perm][:, perm]."""
    e, _, perm = R.graded_errors(est, sig, T, truth_pub)
    perm = list(perm)
    return np.asarray(e, float), np.asarray(P, float)[np.ix_(perm, perm)]


# ------------------------------------------------------------------------------------------------ rooms and the first stay
def read_gt_pose(run):
    out = []
    for r in R.read_tsv(run / 'gt_pose.tsv'):
        try: out.append((int(r['t_ms']), float(r['x']), float(r['y'])))
        except (KeyError, ValueError, TypeError): continue
    out.sort()
    return out


def room_series(samples, polys):
    """[(t, room name or None)]: the room polygon each Webots sample lies in (None: in no polygon, e.g. a doorway)."""
    return [(t, next((k for k, p in polys.items() if G._point_in(p, (x, y))), None)) for t, x, y in samples]


def first_stay(series, room):
    """(t_in, t_out, open) of the robot's first stay in `room`: t_in = the first sample inside it, t_out = the LAST sample inside
    it before the first sample in a different room (None samples are neutral). open = the log ended inside the room. None when
    the robot was never inside."""
    t_in = t_out = None
    for t, r in series:
        if r == room:
            if t_in is None: t_in = t
            t_out = t
        elif r is not None and t_in is not None:
            return t_in, t_out, False
    return None if t_in is None else (t_in, t_out, True)


# ------------------------------------------------------------------------------------------------ the replay (full P)
def replay_rows(dump, replay_bin, cache):
    """({upd: row dict}, header ok, exit code, stderr) of the replay tool run on `dump` with --theta-cov, cached by dump identity."""
    st = os.stat(dump)
    key = hashlib.sha1(f'{dump}|{st.st_size}|{st.st_mtime_ns}|{replay_bin}|{os.stat(replay_bin).st_mtime_ns}'.encode()).hexdigest()[:16]
    csvp, errp, rcp = (Path(cache) / f'{Path(dump).stem}_{key}.{x}' for x in ('csv', 'err', 'rc'))
    if not (csvp.exists() and errp.exists() and rcp.exists()):
        Path(cache).mkdir(parents=True, exist_ok=True)
        with open(csvp.with_suffix('.csv.tmp'), 'w') as fo, open(errp, 'w') as fe:
            rc = subprocess.run([replay_bin, str(dump), '--theta-cov'], stdout=fo, stderr=fe).returncode
        os.replace(csvp.with_suffix('.csv.tmp'), csvp); rcp.write_text(str(rc))
    with open(csvp) as fh: rows = list(csv.DictReader(fh))
    return rows, int(rcp.read_text()), errp.read_text()


def dump_for(run, node):
    d = run / 'replay'
    if not d.exists(): return []
    import re
    return sorted(p for p in d.iterdir() if re.fullmatch(re.escape(node) + r'_\d+_\d+\.bin', p.name))


# ------------------------------------------------------------------------------------------------ one tour
def instances(wbt):
    """[(key, name, cls)]: every instance of the seven classes (key = its DEF, else its Webots name)."""
    import mesh_extents as M
    out = []
    for d, nm, c, _ in M.world_nodes(wbt):
        if c in CLASSES: out.append((d or nm, nm, c))
    return out


def score_tour(run, wbt, replay_bin, cache, smoke=False, only=None, say=print):
    """[dict]: one record per (instance) of the tour: status 'ok' | 'unmatched' | 'no full P' | 'not visited', and the numbers."""
    run = R.run_path(run)
    series = room_series(read_gt_pose(run), G.room_polygons())
    rows_by_cls, cycles_by_cls, recs = {}, {}, []
    for key, name, cls in instances(wbt):
        if only and cls not in only: continue
        rec = dict(run=run.name, key=key, name=name, cls=cls, status='', why='')
        recs.append(rec)
        T = G.object_truth(wbt, key)
        rec['room'] = T['room_poly']
        stay = first_stay(series, T['room_poly'])
        if stay is None: rec.update(status='not visited', why=f'the robot was never inside {T["room_poly"]}'); continue
        t_in, t_out, is_open = stay
        rec.update(t_in=t_in, t_out=t_out, stay_open=is_open)
        rows = rows_by_cls.setdefault(cls, R.ai2_rows(run, cls))
        S = {'A1': {'t_claim_ms': t_in, 't_dwell1_ms': t_out}, 'A2': {'t_dwell1_ms': t_out}}
        m = R.match_node_to_truth(run, rows, S, T)
        rec.update(d2=m['d2'], runner=m['runner'], runner_d2=m['runner_d2'], fit_rms=m['fit_rms'], fit_n=m['fit_n'])
        if m['node'] is None: rec.update(status='unmatched', why=m['why'] or 'no node'); continue
        node = m['node']; fr = m['frames']; rec['node'] = node
        ups = R.updates(rows, node)
        inside = [u for u in ups if u['t'] <= t_out]
        if not inside: rec.update(status='no full P', why=f'{node} has no update up to the end of the stay'); continue
        k = len(inside) - 1; d = inside[k]
        rec.update(t_first=inside[0]['t'], t_draw=d['t'], n_updates=len(inside))
        px, py, pyaw = fr.world_to_pub(T['cx'], T['cy'], T['yaw_theta_world'], t_out)
        est, sig = d['est'], d['sig']
        if T.get('symmetry') == 'c2v' or cls == 'table': rec['label'] = 'old engine, diagonal'
        # --- full P from the replay
        dumps = dump_for(run, node)
        P = why = None
        if len(dumps) == 1:
            rr, rc, err = replay_rows(dumps[0], replay_bin, cache)
            if k >= len(rr): why = f'dump has {len(rr)} updates, the ai2 log {len(ups)}'
            else:
                r = rr[k]
                cyc = cycles_by_cls.setdefault(cls, update_cycles(run, cls)).get(node, [])
                same = (len(cyc) == len(ups) and int(r['cycle']) == cyc[k] and int(r['gated']) == d['gated']
                        and int(r['lidar_only']) == d['lidar_only'])
                dth, dP = float(r['dth_live']), float(r['dP_live'])
                rec.update(dth_live=dth, dP_live=dP, replay_rc=rc, dtheta_vs_ai2=max(abs(float(r[c]) - e) for c, e in zip(DOFS, est)))
                if not same: why = f'update {k}: replay cycle/gated/lidar_only differ from the ai2 log'
                elif not smoke and rc != 0: why = f'replay tool exit {rc}: ' + (err.strip().splitlines() or ['?'])[-1][:120]
                elif not smoke and not (dth == 0.0 and dP == 0.0): why = f'replay not bit-identical to the live run at the draw (dth_live {dth:.3g}, dP_live {dP:.3g})'
                else:
                    est = [float(r[c]) for c in DOFS]; sig = [math.sqrt(float(r[f'S_{i}_{i}'])) for i in range(6)]; P = cov_from_row(r)
        else:
            why = f'{len(dumps)} replay dumps for {node}' if dumps else f'no replay dump for {node}'
        if P is None and cls == 'table':
            rec.update(status='ok', why=why or '', label='old engine, diagonal')       # legacy: diagonal from the ai2 log sigma
            e, s, _ = R.graded_errors(d['est'], d['sig'], T, (px, py, pyaw))
            q, perdof = nees_diag(e, s)
        elif P is None:
            rec.update(status='no full P', why=why); continue
        else:
            e, Pg = graded_error_full(est, sig, P, T, (px, py, pyaw))
            try: q, perdof = nees_full(e, Pg)
            except np.linalg.LinAlgError: rec.update(status='no full P', why='theta covariance not positive definite'); continue
            rec.update(status='ok', why='')
        rec.update(nees=q, perdof=[float(x) for x in perdof], e=[float(x) for x in e])
    return recs


def update_cycles(run, cls):
    """{node: [ai2 `cycle` of each update, in aba_runlog.updates order]}: aba_runlog.ai2_rows drops that column, so the file is read
    again with the same update rule (camera: gated 0, lidar_only 0, npts > 0; LiDAR: beam_qsum > 0 and (gated 0 or lidar_only 1))
    and the same stable sort by pkt_ts. The caller checks the count against len(updates)."""
    out = []
    with open(run / f'{cls}_ai2_log.csv') as fh:
        for r in csv.DictReader(fh):
            try:
                g, lo, n, q = int(float(r['gated'])), int(float(r['lidar_only'])), int(float(r['npts'])), float(r['beam_qsum'])
                t, cyc, node = int(r['pkt_ts']), int(float(r['cycle'])), r['node']
                float(r['rho_cam']); [float(r[k]) for k in R.SIG + R.EST]   # the rows ai2_rows would skip are skipped here too
            except (KeyError, ValueError):
                continue
            if (g == 0 and lo == 0 and n > 0) or (q > 0 and (g == 0 or lo == 1)): out.append((t, node, cyc))
    out.sort(key=lambda x: x[0])
    d = {}
    for _, node, cyc in out: d.setdefault(node, []).append(cyc)
    return d


# ------------------------------------------------------------------------------------------------ the verdict per class
def bootstrap_ci(x, n=BOOT_N, seed=BOOT_SEED):
    x = np.asarray(x, float)
    if len(x) == 0: return (math.nan, math.nan)
    idx = np.random.default_rng(seed).integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summarise(recs, cls):
    rs = [r for r in recs if r['cls'] == cls and r['status'] != 'not visited']
    ok = [r for r in rs if r['status'] == 'ok']
    q = [r['nees'] for r in ok]
    mean = float(np.mean(q)) if q else math.nan
    lo, hi = bootstrap_ci(q)
    perdof = np.mean([r['perdof'] for r in ok], 0).tolist() if ok else [math.nan] * 6
    n_un = sum(r['status'] == 'unmatched' for r in rs); n_nop = sum(r['status'] == 'no full P' for r in rs)
    why = []
    if len(ok) < N_MIN: why.append(f'n graded {len(ok)} < {N_MIN}')
    if n_un: why.append(f'{n_un} unmatched')
    if n_nop: why.append(f'{n_nop} no full P')
    if not (PASS_LO <= mean <= PASS_HI): why.append(f'mean {mean:.2f} outside [{PASS_LO}, {PASS_HI}]')
    labels = sorted({r.get('label', '') for r in ok} - {''})
    return dict(cls=cls, visits=len(rs), matched=sum(r['status'] != 'unmatched' for r in rs), graded=len(ok), mean=mean, lo=lo, hi=hi,
                perdof=perdof, passed=not why, why=why, label='; '.join(labels), not_visited=sum(r['cls'] == cls and r['status'] == 'not visited' for r in recs))


def report(recs, classes=CLASSES, say=print):
    say('first-visit NEES/DOF, 6-DOF, full theta covariance (AuRo test a)   registered pass: n >= 10 and mean in [0.5, 2]')
    for c in classes:
        s = summarise(recs, c)
        if s['visits'] == 0 and not s['not_visited']: continue
        say(f"{c:13s} visits {s['visits']:3d}  matched {s['matched']:3d}  graded {s['graded']:3d}  mean NEES/DOF {s['mean']:7.3f}  "
            f"95% CI [{s['lo']:.3f}, {s['hi']:.3f}]  {'PASS' if s['passed'] else 'FAIL'}"
            + (f"  [{s['label']}]" if s['label'] else '') + (f"  ({'; '.join(s['why'])})" if s['why'] else '')
            + (f"  not visited {s['not_visited']}" if s['not_visited'] else ''))
        say('              per-DOF mean e^2/P_kk  ' + '  '.join(f'{d} {v:.3f}' for d, v in zip(DOFS, s['perdof'])))
    say('--- per visit')
    for r in recs:
        if classes and r['cls'] not in classes: continue
        base = f"{r['run']} {r['cls']:12s} {r['key']:24s} {r['room']:7s} {r['status']:11s}"
        if r['status'] == 'ok':
            say(f"{base} node {r['node']} d2 {r['d2']:.2f} nees/DOF {r['nees']:.3f} (n_upd {r['n_updates']}, dtheta_vs_ai2 {r.get('dtheta_vs_ai2', math.nan):.2g})"
                + (f" [{r['label']}]" if r.get('label') else '') + (f" ({r['why']})" if r['why'] else ''))
        else:
            say(f"{base} {r['why']}" + (f" (nearest d2 {r['d2']:.1f})" if r['status'] == 'unmatched' and 'd2' in r else ''))


def main(argv):
    wbt, replay_bin, cache, smoke, only, runs = DEFAULT_WBT, DEFAULT_REPLAY, os.path.expanduser('~/.cache/first_visit_nees'), False, None, []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == '--wbt': wbt = argv[i + 1]; i += 1
        elif a == '--replay': replay_bin = argv[i + 1]; i += 1
        elif a == '--cache': cache = argv[i + 1]; i += 1
        elif a == '--classes': only = set(argv[i + 1].split(',')); i += 1
        elif a == '--smoke': smoke = True
        else: runs.append(a)
        i += 1
    if not runs: print(__doc__); return 2
    recs = []
    for run in runs: recs += score_tour(run, wbt, replay_bin, cache, smoke, only)
    report(recs, tuple(c for c in CLASSES if not only or c in only))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
