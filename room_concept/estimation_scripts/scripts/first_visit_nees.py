#!/usr/bin/env python3
"""AuRo test (a), shared-engine part 1 grading: first-visit 6-DOF full-covariance NEES over tours.
    python3 -I first_visit_nees.py <tour_run_dir>... [--wbt W] [--replay BIN] [--cache DIR] [--smoke] [--classes a,b]
                                    [--counterfactual] [--yaw-var=reported]
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
  table       the legacy binary writes no dumps (and a different ai2 header: no lidar_only / beam_qsum / rho_* / spose_* / lever_*, so
              aba_runlog.ai2_rows reads none of it; legacy_rows reads it with the legacy log's own rule: an update is a row with
              gated == 0 and npts > 0, as table_fitter.cpp writes npts = 0 on the no-mask rows). Match and draw use those rows. With NO
              dump the NEES is DIAGONAL from the log's sigma, labelled "old engine, diagonal". A table WITH a dump is graded with full P
              like any class (the c2v image chosen by the full-P NEES); a dump that fails validation is "no full P", never diagonal.
              Any other class without a usable dump: the visit is "no full P", never silently diagonal.
  P and fit    P is the node's own posterior; fit_rms^2 never enters it. fit_rms per visit and per-class mean fit_rms^2 against mean
              P_cx, P_cy are PRINTED (report only, not in the verdict).
  duplicates  one node matched to two instances of a tour: both are "duplicate", fail the class, and give no draw.
  no room     an instance whose centre is in no room polygon is skipped, listed as "no room" (not a visit).
  output      per class: visits, matched, graded, mean NEES/DOF, bootstrap 95 % CI over visits (percentile, 10000 resamples of the
              graded visits, fixed seed: a computation, not a bound), per-DOF mean e^2/P_kk, PASS/FAIL. No other statistic, no
              outlier removal. PASS needs: n graded >= 10, no unmatched visit, no 'no full P' visit, mean in [0.5, 2].
Every number is read with python float()/int() or struct-free text parsing, which does not read the locale.

Opt-in, NOT the registered grade (task-F1; the default output is unchanged):
  --counterfactual     grade a COUNTERFACTUAL replay (a replay binary that re-stages the frames with a changed rule, e.g. the replay
                       tool's --f1-restage): skips ONLY the bit-identity gate (dth_live = dP_live = 0), which correctly rejects such a
                       replay; every other gate stays. Every output line is prefixed "COUNTERFACTUAL replay — not the registered grade".
  --yaw-var=reported   grade the engine's PUBLISHED covariance: the full 6x6 theta block with its yaw diagonal replaced by the
                       reported marginal yaw variance (the replay's std_yaw column squared = yaw_marginal_var(), the S(5,5) of
                       covariance_reported()), recomputed as a FULL-covariance NEES. The registered statistic uses the within-mode
                       Sigma(5,5). Every output line is prefixed "engine-reported yaw variance — NOT the registered statistic".
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
    """(e, P in the TRUTH's DOF labels). The error is aba_runlog.graded_errors. For a c2v class (a table) the four equivalent
    descriptions (aba_runlog.c2v_images) are each scored with graded_errors on that image alone, P permuted with the image
    (perm[k] = the estimate's DOF that stands for DOF k: P_img = P[perm][:, perm]), and the image with the smallest FULL-P NEES is
    returned (graded_errors' own pick uses a diagonal cost, which is not this statistic)."""
    P = np.asarray(P, float)
    if T.get('symmetry') != 'c2v':
        e, _, _ = R.graded_errors(est, sig, T, truth_pub)
        return np.asarray(e, float), P
    best = None
    for e_, s_, perm in R.c2v_images(est, sig):
        e, _, _ = R.graded_errors(e_, s_, dict(T, symmetry='none'), truth_pub)
        perm = list(perm); Pi = P[np.ix_(perm, perm)]
        try: q = nees_full(e, Pi)[0]
        except np.linalg.LinAlgError: continue
        if best is None or q < best[0]: best = (q, np.asarray(e, float), Pi)
    if best is None: raise np.linalg.LinAlgError('no c2v image has a positive definite P')
    return best[1], best[2]


# ------------------------------------------------------------------------------------------------ rooms and the first stay
def legacy_rows(run, cls):
    """The ai2 log of a LEGACY binary (table_concept, table_fitter.cpp log_ai2_csv): header without lidar_only / beam_qsum / rho_* /
    spose_* / lever_*. Rows shaped like aba_runlog.ai2_rows' (node, t, est, sig, cam, L ...); an update = gated == 0 and npts > 0 (the
    log writes npts = 0 on the no-mask rows, and gated cycles do not update); the legacy engine has no separate LiDAR-only update."""
    p = run / f'{cls}_ai2_log.csv'
    if not p.exists(): return []
    out = []
    with open(p) as fh:
        for r in csv.DictReader(fh):
            try:
                d = dict(node=r['node'], t=int(r['pkt_ts']), gated=int(float(r['gated'])), npts=int(float(r['npts'])), lidar_only=0,
                         beam_qsum=0.0, rho_cam=1.0, rho_L=1.0, rho_chain=1.0,
                         sig=[float(r[k]) for k in R.SIG], est=[float(r[k]) for k in R.EST], cycle=int(float(r['cycle'])))
            except (KeyError, ValueError):
                continue
            d['cam'] = d['gated'] == 0 and d['npts'] > 0
            d['L'] = False
            out.append(d)
    out.sort(key=lambda d: d['t'])
    return out


def class_rows(run, cls):
    """(rows, legacy): aba_runlog.ai2_rows, else the legacy reader when the file has the legacy header."""
    rows = R.ai2_rows(run, cls)
    if rows: return rows, False
    p = run / f'{cls}_ai2_log.csv'
    if p.exists():
        with open(p) as fh: hdr = fh.readline().strip().split(',')
        if 'lidar_only' not in hdr: return legacy_rows(run, cls), True
    return [], False


def polygons(wbt):
    """The room polygons of THIS world file (wbt_rooms reads its module default otherwise), and aba_geometry made to use them."""
    import wbt_rooms as W
    if os.path.abspath(wbt) != os.path.abspath(W.WBT):
        W.WBT = wbt; W.MESH_DIR = os.path.join(os.path.dirname(wbt), 'meshes'); W.parse_wbt.__defaults__ = (wbt,)
    polys = G.room_polygons()
    G.room_polygons = lambda: polys
    return polys


def replay_identity(replay_bin):
    """One line: the replay binary, its mtime, the HEAD of the source tree it sits in, and whether any source it is built from
    (countable_concept/src, tools/replay, common/ai_belief, common/concept_manifest) is newer than the binary."""
    import datetime
    try: st = os.stat(replay_bin)
    except OSError: return f'replay binary {replay_bin}: NOT FOUND'
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(replay_bin)), '..', '..', '..'))
    def git(*a):
        r = subprocess.run(['git', '-C', root, *a], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else '?'
    newest = 0.0
    for d in ('countable_concept/src', 'countable_concept/tools/replay', 'common/ai_belief', 'common/concept_manifest'):
        for dp, _, fs in os.walk(os.path.join(root, d)):
            for f in fs:
                if f.endswith(('.h', '.cpp')): newest = max(newest, os.path.getmtime(os.path.join(dp, f)))
    return (f"replay binary {replay_bin}  built {datetime.datetime.fromtimestamp(st.st_mtime):%Y-%m-%d %H:%M}  source tree {root} "
            f"HEAD {git('describe', '--always', '--dirty')} ({git('log', '-1', '--format=%cd', '--date=iso')})  "
            + ('STALE: a source file is newer than the binary' if newest > st.st_mtime else 'no source file newer than the binary'))


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


def score_tour(run, wbt, replay_bin, cache, smoke=False, only=None, runner=None, say=print, counterfactual=False, yaw_var='within'):
    """[dict]: one record per instance of the tour: status 'ok' | 'unmatched' | 'no full P' | 'duplicate' | 'not visited' | 'no room'.
    runner(dump, replay_bin, cache) -> (rows, rc, stderr) is injectable (tests). counterfactual / yaw_var: the opt-in statistics of
    the module docstring (defaults = the registered grade)."""
    runner = runner or replay_rows
    run = R.run_path(run)
    series = room_series(read_gt_pose(run), polygons(wbt))
    rows_by_cls, cycles_by_cls, recs = {}, {}, []
    for key, name, cls in instances(wbt):
        if only and cls not in only: continue
        rec = dict(run=run.name, key=key, name=name, cls=cls, status='', why='', room='?')
        recs.append(rec)
        try: T = G.object_truth(wbt, key)
        except ValueError as ex: rec.update(status='no room', why=str(ex)); continue
        rec['room'] = T['room_poly']
        stay = first_stay(series, T['room_poly'])
        if stay is None: rec.update(status='not visited', why=f'the robot was never inside {T["room_poly"]}'); continue
        t_in, t_out, is_open = stay
        rec.update(t_in=t_in, t_out=t_out, stay_open=is_open)
        if cls not in rows_by_cls: rows_by_cls[cls] = class_rows(run, cls)
        rows, legacy = rows_by_cls[cls]
        S = {'A1': {'t_claim_ms': t_in, 't_dwell1_ms': t_out}, 'A2': {'t_dwell1_ms': t_out}}
        m = R.match_node_to_truth(run, rows, S, T)
        rec.update(d2=m['d2'], runner=m['runner'], runner_d2=m['runner_d2'], fit_rms=m['fit_rms'], fit_n=m['fit_n'], legacy_log=legacy)
        if m['node'] is None: rec.update(status='unmatched', why=m['why'] or 'no node'); continue
        node = m['node']; fr = m['frames']; rec['node'] = node
        ups = R.updates(rows, node)
        inside = [u for u in ups if u['t'] <= t_out]
        if not inside: rec.update(status='no full P', why=f'{node} has no update up to the end of the stay'); continue
        k = len(inside) - 1; d = inside[k]
        rec.update(t_first=inside[0]['t'], t_draw=d['t'], n_updates=len(inside))
        px, py, pyaw = fr.world_to_pub(T['cx'], T['cy'], T['yaw_theta_world'], t_out)
        truth_pub = (px, py, pyaw)
        dumps = dump_for(run, node)
        P = None; why = ''; est = sig = None
        if len(dumps) == 1:
            rr, rc, err = runner(dumps[0], replay_bin, cache)
            if len(rr) != len(ups): why = f'the dump has {len(rr)} updates, the ai2 log {len(ups)} for {node}'
            else:
                r = rr[k]
                if legacy: cyc_k, ok_cyc = d['cycle'], True
                else:
                    if cls not in cycles_by_cls: cycles_by_cls[cls] = update_cycles(run, cls)
                    cyc = cycles_by_cls[cls].get(node, []); ok_cyc = len(cyc) == len(ups); cyc_k = cyc[k] if ok_cyc else None
                if not ok_cyc: why = 'the ai2 cycle column does not align with its updates'
                else:
                    dth, dP = float(r['dth_live']), float(r['dP_live'])
                    rec.update(dth_live=dth, dP_live=dP, replay_rc=rc, dtheta_vs_ai2=max(abs(float(r[c]) - e) for c, e in zip(DOFS, d['est'])))
                    if not (int(r['upd']) == k and int(r['cycle']) == cyc_k and int(r['gated']) == d['gated'] and int(r['lidar_only']) == d['lidar_only']):
                        why = f'update {k}: replay upd/cycle/gated/lidar_only differ from the ai2 log'
                    elif not smoke and rc != 0: why = f'replay tool exit {rc}: ' + (err.strip().splitlines() or ['?'])[-1][:120]
                    elif not smoke and not counterfactual and not (dth == 0.0 and dP == 0.0): why = f'replay not bit-identical to the live run at the draw (dth_live {dth:.3g}, dP_live {dP:.3g})'
                    else:
                        est = [float(r[c]) for c in DOFS]; P = cov_from_row(r)
                        if yaw_var == 'reported':
                            rec['yaw_var_within'] = float(P[5, 5]); P[5, 5] = float(r['std_yaw']) ** 2
                        sig = [math.sqrt(P[i, i]) for i in range(6)]
        else:
            why = f'{len(dumps)} replay dumps for {node}' if dumps else f'no replay dump for {node}'
        if P is None and cls == 'table' and not dumps:
            e, s_, _ = R.graded_errors(d['est'], d['sig'], T, truth_pub)       # the legacy binary: no dump exists; diagonal from the log
            q, perdof = nees_diag(e, s_)
            rec.update(label='old engine, diagonal', pcx=float(d['sig'][0]) ** 2, pcy=float(d['sig'][1]) ** 2)
        elif P is None:
            rec.update(status='no full P', why=why); continue
        else:
            try:
                e, Pg = graded_error_full(est, sig, P, T, truth_pub); q, perdof = nees_full(e, Pg)
            except np.linalg.LinAlgError: rec.update(status='no full P', why='theta covariance not positive definite'); continue
            rec.update(label='', pcx=float(Pg[0, 0]), pcy=float(Pg[1, 1]))
        rec.update(status='ok', why='', nees=q, perdof=[float(x) for x in perdof], e=[float(x) for x in e])
    # one node matched to two instances of the tour: a duplicate draw on both, failing both
    by_node = {}
    for r in recs:
        if r['status'] in ('ok', 'no full P') and r.get('node'): by_node.setdefault((r['cls'], r['node']), []).append(r)
    for (cls, node), rs in by_node.items():
        if len(rs) > 1:
            for r in rs:
                r.update(status='duplicate', why=f'{node} is the match of {len(rs)} instances: ' + ', '.join(x['key'] for x in rs))
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
    rs = [r for r in recs if r['cls'] == cls and r['status'] not in ('not visited', 'no room')]
    ok = [r for r in rs if r['status'] == 'ok']
    q = [r['nees'] for r in ok]
    mean = float(np.mean(q)) if q else math.nan
    lo, hi = bootstrap_ci(q)
    perdof = np.mean([r['perdof'] for r in ok], 0).tolist() if ok else [math.nan] * 6
    n_un = sum(r['status'] == 'unmatched' for r in rs); n_nop = sum(r['status'] == 'no full P' for r in rs)
    n_dup = sum(r['status'] == 'duplicate' for r in rs)
    why = []
    if len(ok) < N_MIN: why.append(f'n graded {len(ok)} < {N_MIN}')
    if n_un: why.append(f'{n_un} unmatched')
    if n_nop: why.append(f'{n_nop} no full P')
    if n_dup: why.append(f'{n_dup} duplicate draw')
    if not (PASS_LO <= mean <= PASS_HI): why.append(f'mean {mean:.2f} outside [{PASS_LO}, {PASS_HI}]')
    labels = sorted({r.get('label', '') for r in ok} - {''})
    fit = [r for r in ok if r.get('fit_rms') is not None and math.isfinite(r['fit_rms'])]
    frame = dict(fit_rms2=float(np.mean([r['fit_rms'] ** 2 for r in fit])) if fit else math.nan,
                 pcx=float(np.mean([r['pcx'] for r in fit])) if fit else math.nan, pcy=float(np.mean([r['pcy'] for r in fit])) if fit else math.nan)
    return dict(cls=cls, visits=len(rs), matched=sum(r['status'] not in ('unmatched',) for r in rs), graded=len(ok), mean=mean, lo=lo, hi=hi,
                perdof=perdof, passed=not why, why=why, label='; '.join(labels), frame=frame,
                not_visited=sum(r['cls'] == cls and r['status'] == 'not visited' for r in recs),
                no_room=sum(r['cls'] == cls and r['status'] == 'no room' for r in recs))


def report(recs, classes=CLASSES, say=print):
    say('first-visit NEES/DOF, 6-DOF, full theta covariance (AuRo test a)   registered pass: n >= 10 and mean in [0.5, 2]')
    for c in classes:
        s = summarise(recs, c)
        if s['visits'] == 0 and not s['not_visited'] and not s['no_room']: continue
        say(f"{c:13s} visits {s['visits']:3d}  matched {s['matched']:3d}  graded {s['graded']:3d}  mean NEES/DOF {s['mean']:7.3f}  "
            f"95% CI [{s['lo']:.3f}, {s['hi']:.3f}]  {'PASS' if s['passed'] else 'FAIL'}"
            + (f"  [{s['label']}]" if s['label'] else '') + (f"  ({'; '.join(s['why'])})" if s['why'] else '')
            + (f"  not visited {s['not_visited']}" if s['not_visited'] else '') + (f"  no room {s['no_room']}" if s['no_room'] else ''))
        say('              per-DOF mean e^2/P_kk  ' + '  '.join(f'{d} {v:.3f}' for d, v in zip(DOFS, s['perdof'])))
        f = s['frame']
        say(f"              frame fit (report only): mean fit_rms^2 {f['fit_rms2']:.2e} m^2 vs mean P_cx {f['pcx']:.2e}, P_cy {f['pcy']:.2e} m^2")
        if c == 'bed': say('              note: h inside the interval contributes 0 to e; the mean is over 6 DOF as registered')
    say('--- per visit')
    for r in recs:
        if classes and r['cls'] not in classes: continue
        base = f"{r['run']} {r['cls']:12s} {r['key']:24s} {r['room']:7s} {r['status']:11s}"
        if r['status'] == 'ok':
            say(f"{base} node {r['node']} d2 {r['d2']:.2f} fit_rms {r['fit_rms']:.4f} m nees/DOF {r['nees']:.3f} (n_upd {r['n_updates']}, dtheta_vs_ai2 {r.get('dtheta_vs_ai2', math.nan):.2g})"
                + (f" [{r['label']}]" if r.get('label') else '') + (' [legacy log]' if r.get('legacy_log') else ''))
        else:
            say(f"{base} {r['why']}" + (f" (nearest d2 {r['d2']:.1f})" if r['status'] == 'unmatched' and 'd2' in r else ''))


COUNTERFACTUAL_LABEL = 'COUNTERFACTUAL replay — not the registered grade'
REPORTED_YAW_LABEL = 'engine-reported yaw variance — NOT the registered statistic'


def labelled(say, counterfactual, yaw_var):
    """say() with the opt-in statistics' labels prefixed to every line (identity when both are off)."""
    tags = ([COUNTERFACTUAL_LABEL] if counterfactual else []) + ([REPORTED_YAW_LABEL] if yaw_var == 'reported' else [])
    if not tags: return say
    pre = '[' + '; '.join(tags) + '] '
    return lambda line: say(pre + line)


def main(argv):
    wbt, replay_bin, cache, smoke, only, runs = DEFAULT_WBT, DEFAULT_REPLAY, os.path.expanduser('~/.cache/first_visit_nees'), False, None, []
    counterfactual, yaw_var = False, 'within'
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == '--counterfactual': counterfactual = True
        elif a.startswith('--yaw-var='):
            yaw_var = a.split('=', 1)[1]
            if yaw_var not in ('within', 'reported'): print(f'--yaw-var must be within or reported, not {yaw_var}'); return 2
        elif a == '--wbt': wbt = argv[i + 1]; i += 1
        elif a == '--replay': replay_bin = argv[i + 1]; i += 1
        elif a == '--cache': cache = argv[i + 1]; i += 1
        elif a == '--classes': only = set(argv[i + 1].split(',')); i += 1
        elif a == '--smoke': smoke = True
        else: runs.append(a)
        i += 1
    if not runs: print(__doc__); return 2
    say = labelled(print, counterfactual, yaw_var)
    say(replay_identity(replay_bin))
    recs = []
    for run in runs: recs += score_tour(run, wbt, replay_bin, cache, smoke, only, counterfactual=counterfactual, yaw_var=yaw_var)
    report(recs, tuple(c for c in CLASSES if not only or c in only), say=say)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
