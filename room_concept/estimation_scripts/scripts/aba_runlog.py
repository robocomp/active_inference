#!/usr/bin/env python3
"""Run-folder reading shared by parked_report.py and aba_report.py (shared-engine plan rev 2, Task P1).

A run folder (crossing/<run>) made by parked_test.sh holds, besides the tour's files:
  aba_segments.tsv          one row per scripted segment (A1, B, A2) of each object: claim / arrival / dwell times (wall ms)
  aba_promises.tsv          the object's last published offer at each claim (offers.csv rows read live by aba_visit.sh)
  aba_plan_<DEF>.json       the standpoints and the object's ground truth (aba_geometry.py plan)
  <class>_ai2_log.csv, <class>_offers.csv, <class>_delivered.csv   harvested by concept_agents.sh
  gt_error.csv, proto_room.csv, gt_pose.tsv, agents.txt
Every number is parsed with python float()/int(), which do not read the locale.

UPDATES. An ai2 row is an update of a channel when (countable_fitter.cpp, the gated path):
  camera: gated == 0, lidar_only == 0 and npts > 0;      LiDAR: beam_qsum > 0 and (gated == 0 or lidar_only == 1).
Its rho_cam / rho_L / rho_chain are the lag-1 correlations that update used (1 when the channel did not measure). The time is
pkt_ts (the capture stamp, epoch ms, the clock of gt_pose.tsv and of aba_segments.tsv).
"""
import csv, json, math, os
from pathlib import Path

import aba_model as Mdl

SIG = ('std_cx', 'std_cy', 'std_H', 'std_w', 'std_h', 'std_yaw_within')   # yaw: the engine's within-mode Σ(5,5)
EST = ('cx', 'cy', 'H', 'w', 'h', 'yaw')


def run_path(arg):
    """A folder, else a run name under main's crossing/ (where the tour scripts write, from any checkout), else this one's."""
    for p in (Path(arg), Path('/home/pbustos/robocomp/components/active_inference/room_concept/estimation_scripts/crossing') / arg,
              Path(__file__).resolve().parent.parent / 'crossing' / arg):
        if p.exists(): return p.resolve()
    return Path(arg).resolve()


def read_tsv(path):
    if not Path(path).exists(): return []
    with open(path) as fh: return list(csv.DictReader(fh, delimiter='\t'))


def read_csv(path):
    if not Path(path).exists(): return []
    with open(path) as fh: return list(csv.DictReader(fh))


SEGS = ('A1', 'B', 'A2')
TIMES = ('t_claim_ms', 't_arrive_ms', 't_dwell0_ms', 't_dwell1_ms')


def segments(run, report=print):
    """{DEF: {seg: dict}} from aba_segments.tsv (empty without it). A malformed row (a missing column, a time that is not an
    integer) is SKIPPED and said so: its segment is then missing, and visit_validity gives the object NO VERDICT."""
    out = {}
    for i, r in enumerate(read_tsv(run / 'aba_segments.tsv'), 2):
        try:
            d = dict(r)
            if any(v is None for v in d.values()) or None in d: raise ValueError('wrong number of columns')
            for k in TIMES: d[k] = int(r[k])
            out.setdefault(r['def'], {})[r['seg']] = d
        except (KeyError, ValueError, TypeError) as e:
            report(f'aba_segments.tsv line {i}: malformed ({e}) — skipped')
    return out


def controller_goal_tolerance(root='/home/pbustos/robocomp/components/active_inference'):
    """The controller's arrival radius (m) and where it was read: a Controller.GoalThreshold key in controller/etc/config.toml
    if the config sets one, else the compiled default `goal_threshold` in controller/src/controller_types.h — the value the
    controller judges arrival with (the config sets none today; mood scaling of it is off by design)."""
    import re
    cfg = os.path.join(root, 'controller', 'etc', 'config.toml')
    if os.path.exists(cfg):
        for line in open(cfg):
            m = re.match(r'\s*GoalThreshold\s*=\s*([-+0-9.eE]+)', line)
            if m: return float(m.group(1)), f'{cfg} GoalThreshold'
    hdr = os.path.join(root, 'controller', 'src', 'controller_types.h')
    for line in open(hdr):
        m = re.match(r'\s*float\s+goal_threshold\s*=\s*([-+0-9.eE]+)f?\s*;', line)
        if m: return float(m.group(1)), f'{hdr} goal_threshold (no config key)'
    raise KeyError('controller goal tolerance not found')


def visit_validity(S, tol_m):
    """[] when the visit happened as scripted, else the reasons it did not: all three segments present, goto / dwell completed,
    face completed or none, both gain queries answered, and the Webots arrival within the controller's goal tolerance of the
    scripted standpoint."""
    why = []
    for s in SEGS:
        if s not in S: why.append(f'{s}: segment missing'); continue
        r = S[s]
        if r.get('goto') != 'completed': why.append(f'{s}: goto {r.get("goto")}')
        # face = completed[attempts...] (aba_visit.sh face_step), or the plain 'completed' / 'none' of earlier runs
        f = r.get('face') or ''
        if not (f in ('completed', 'none') or f.startswith('completed[')): why.append(f'{s}: face {f}')
        if r.get('dwell') != 'completed': why.append(f'{s}: dwell {r.get("dwell")}')
        if r.get('query_answered') != 'yes': why.append(f'{s}: claim gain query {r.get("query_answered")}')
        if r.get('arr_query_answered') != 'yes': why.append(f'{s}: arrival gain query {r.get("arr_query_answered")}')
        try:
            d = math.hypot(float(r['gt_x']) - float(r['stand_x']), float(r['gt_y']) - float(r['stand_y']))
        except (KeyError, ValueError, TypeError):
            d = math.nan
        if not d <= tol_m: why.append(f'{s}: arrival {d:.3f} m from the standpoint (tolerance {tol_m:.3f} m)')
    return why


def checkout_root(run):
    """The checkout the countable agents RAN from (agents.txt: <checkout>/<agent>/bin/<agent>), for its manifests/configs;
    else the checkout this script lives in."""
    for line in (run / 'agents.txt').read_text().splitlines() if (run / 'agents.txt').exists() else []:
        f = line.split('\t')
        if len(f) >= 2 and f[0] in ('nightstand_concept', 'bed_concept', 'toilet_concept'):
            return str(Path(f[1]).parent.parent.parent)
    return str(Path(__file__).resolve().parent.parent.parent.parent)


def ai2_rows(run, cls):
    """[dict] of the class's harvested ai2 log, numeric fields converted, with cam / L update flags."""
    p = run / f'{cls}_ai2_log.csv'
    if not p.exists(): return []
    out = []
    with open(p) as fh:
        for r in csv.DictReader(fh):
            try:
                d = dict(node=r['node'], t=int(r['pkt_ts']), gated=int(float(r['gated'])), npts=int(float(r['npts'])),
                         lidar_only=int(float(r['lidar_only'])), beam_qsum=float(r['beam_qsum']),
                         rho_cam=float(r['rho_cam']), rho_L=float(r['rho_L']), rho_chain=float(r['rho_chain']),
                         sig=[float(r[k]) for k in SIG], est=[float(r[k]) for k in EST],
                         spose=[float(r['spose_x']), float(r['spose_y']), float(r['spose_psi'])],
                         lever=[float(r['lever_x']), float(r['lever_y'])])
            except (KeyError, ValueError):
                continue
            d['cam'] = d['gated'] == 0 and d['lidar_only'] == 0 and d['npts'] > 0
            d['L'] = d['beam_qsum'] > 0 and (d['gated'] == 0 or d['lidar_only'] == 1)
            sx, sy, sp = d['spose']; lx, ly = d['lever']
            d['chain'] = [sx * sx + ly * ly * sp * sp, sy * sy + lx * lx * sp * sp, 0.0, 0.0, 0.0, sp * sp]
            out.append(d)
    out.sort(key=lambda d: d['t'])
    return out


def updates(rows, node):
    return [d for d in rows if d['node'] == node and (d['cam'] or d['L'])]


def match_node(rows, t0, t1):
    """The node with the most updates in [t0, t1] (the object the robot was parked in front of), and that count."""
    n = {}
    for d in rows:
        if t0 <= d['t'] <= t1 and (d['cam'] or d['L']): n[d['node']] = n.get(d['node'], 0) + 1
    return max(n.items(), key=lambda kv: kv[1]) if n else (None, 0)


CHI2_2_99 = -2.0 * math.log(0.01)   # chi-square with 2 dof at 99 %: 9.2103 (the 2-D Gaussian's tail is exp(-d2/2))


def match_node_to_truth(run, rows, S, T):
    """The node that IS the ground-truth object (live run p1_nightstand2: the most-updated node was the OTHER nightstand).
    For every node of the class, its last logged posterior at or before the end of the A1 dwell: its (cx, cy) and their
    sigmas; the truth's centre taken to the published room frame by the visit's frame fit (gt_error over A1 claim .. A2 end,
    the gauge at the A1 dwell end). d2 = sum over x, y of (est - truth)^2 / (sigma^2 + fit_rms^2): the node's own covariance
    (the x-y cross term is not logged) plus the frame fit's error, which is in the comparison too. The match is the smallest d2,
    and only if d2 <= chi2_2(99 %); else no node is the object (NO VERDICT). Returns a dict: node (or None), d2, runner, runner_d2,
    truth (px, py), fit_rms, fit_n, frames, why."""
    from aba_geometry import Frames, read_gauge, read_gt_error
    t1 = S['A1']['t_dwell1_ms']
    out = dict(node=None, d2=math.inf, runner=None, runner_d2=math.inf, truth=None, fit_rms=math.nan, fit_n=0, frames=None, why='')
    try:
        fr = Frames(read_gt_error(run / 'gt_error.csv'), read_gauge(run / 'proto_room.csv'), S['A1']['t_claim_ms'],
                    S['A2']['t_dwell1_ms'])
    except ValueError as e:
        out['why'] = f'no frame fit ({e})'; return out
    px, py, _ = fr.world_to_pub(T['cx'], T['cy'], 0.0, t1)
    out.update(truth=(px, py), fit_rms=fr.rms, fit_n=fr.n, frames=fr)
    cands = []
    for node in sorted({d['node'] for d in rows}):
        d = last_before([u for u in rows if u['node'] == node], t1)
        if d is None: continue
        vx, vy = d['sig'][0] ** 2 + fr.rms ** 2, d['sig'][1] ** 2 + fr.rms ** 2
        cands.append(((d['est'][0] - px) ** 2 / vx + (d['est'][1] - py) ** 2 / vy, node, d))
    cands.sort(key=lambda c: c[0])
    if not cands: out['why'] = 'no node of the class has a posterior before the end of the A1 dwell'; return out
    if len(cands) > 1: out.update(runner=cands[1][1], runner_d2=cands[1][0])
    out['d2'] = cands[0][0]
    if cands[0][0] <= CHI2_2_99: out['node'] = cands[0][1]
    else: out['why'] = f'nearest node {cands[0][1]} at d2 {cands[0][0]:.1f} > chi2_2(99 %) {CHI2_2_99:.2f}'
    out['est'] = cands[0][2]['est'][:2]
    return out


def describe_match(m):
    s = (f"truth ({m['truth'][0]:.3f}, {m['truth'][1]:.3f}) published (fit n={m['fit_n']} rms {m['fit_rms']:.3f} m); "
         if m['truth'] else '')
    s += f"match {m['node'] or 'NONE'} d2 {m['d2']:.2f}" + (f" at ({m['est'][0]:.3f}, {m['est'][1]:.3f})" if 'est' in m else '')
    s += f"; runner-up {m['runner']} d2 {m['runner_d2']:.2f}" if m['runner'] else '; no runner-up'
    return s + f" (bound chi2_2(99 %) = {CHI2_2_99:.2f})" + (f" — {m['why']}" if m['why'] else '')


def last_before(ups, t):
    best = None
    for d in ups:
        if d['t'] <= t: best = d
        else: break
    return best


def _view(ds):
    return dict(cam=any(d['cam'] for d in ds), L=any(d['L'] for d in ds),
                rho_cam=math.prod(d['rho_cam'] for d in ds), rho_L=math.prod(d['rho_L'] for d in ds),
                rho_chain=math.prod(d['rho_chain'] for d in ds),
                chain=[sum(d['chain'][k] for d in ds) / len(ds) for k in range(6)])


def views_lo(ups, t_end):
    """Every update its own view, merged while every logged rho is exactly 1 (an identical shared error: the same view)."""
    groups = []
    for d in ups:
        if d['t'] > t_end: break
        if groups and d['rho_cam'] == 1.0 and d['rho_L'] == 1.0 and d['rho_chain'] == 1.0: groups[-1].append(d)
        else: groups.append([d])
    return [_view(g) for g in groups]


def views_hi(ups, dwells):
    """One view per dwell [(t0, t1)]: its channels = those that measured in it, its rho = the product of every logged rho
    since the previous dwell ended (driving included), its chain = the mean over its updates."""
    out, prev_end = [], -math.inf
    for t0, t1 in dwells:
        inside = [d for d in ups if t0 <= d['t'] <= t1]
        if not inside: prev_end = t1; continue
        between = [d for d in ups if prev_end < d['t'] <= t1]
        v = _view(inside)
        v['rho_cam'] = math.prod(d['rho_cam'] for d in between)
        v['rho_L'] = math.prod(d['rho_L'] for d in between)
        v['rho_chain'] = math.prod(d['rho_chain'] for d in between)
        out.append(v); prev_end = t1
    return out


def predicted_sigmas(views, prm):
    return {h: [math.sqrt(Mdl.theta_var(views, k, prm, h)) for k in range(6)] for h in Mdl.HYPS}


def plan_of(run, defn):
    p = run / f'aba_plan_{defn}.json'
    return json.loads(p.read_text()) if p.exists() else None


def parked_runs(ups):
    """Maximal runs of consecutive updates whose logged rho are all exactly 1 (no view change at all), longest first —
    the parked stretches of a run that has no aba_segments.tsv. Exact equality: rho = exp(-path) is 1 only for a zero path."""
    runs, cur = [], []
    for d in ups:
        if cur and d['rho_cam'] == 1.0 and d['rho_L'] == 1.0 and d['rho_chain'] == 1.0: cur.append(d)
        else:
            if len(cur) > 1: runs.append(cur)
            cur = [d]
    if len(cur) > 1: runs.append(cur)
    return sorted(runs, key=lambda r: r[-1]['t'] - r[0]['t'], reverse=True)
