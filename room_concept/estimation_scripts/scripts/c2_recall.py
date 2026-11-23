#!/usr/bin/env python3
"""AuRo C2 room identity: per room ENTRY of one tour, how the room the robot walked into was decided, and whether it was right.
   c2_recall.py <run_dir> [<run_dir> ...]   -> <run>/c2_entries.csv per run, and a summary over all runs on stdout.

Truth: the Webots track (gt_pose.tsv, 1 Hz), room = the wbt_rooms.py ground-truth polygon containing it (doorways: none, the
previous room holds), named as the graph names them (wbt room_3 = graph room_2, wbt room_2 = graph room_3). An ENTRY is a change of truth room held >= 3 s.
Each log line is timed by gt_pose.tsv's per-log line counts (the first sample whose count reaches the line).

The decision, inside the entry's window [crossing - 5 s, next crossing):
  recognition  room_concept "[room][proposal] ACCEPTED: room <id> is <name>" (ltsm's recogniser proposed, room_concept's fit judged)
  passage      room_concept "[room][recall] <name> via <door>: T_door_recall" (the door crossed is a passage in memory)
  new          neither: the name the controller switched to ("current room changed: a -> b")
  start        (warm) room_concept "MapMode=ltsm: layout of '<name>'" (ltsm's start-up recall)
Expected name: WARM runs (legs.tsv present) remember every room, named by the truth labels above (fixture three_rooms_door0).
EXPLORE runs start empty: the name a truth room got at its FIRST entry is its name; a first entry must be NEW.
  duplicate    a remembered room (re-entry, or any warm entry) that did not get its name
  cut_off      a remembered room left as new only because the fleet stopped while a recall proposal awaited its verdict
  false recall a first entry (explore) given the name of another remembered room
Recogniser numbers per entry: every "[recog] SWAP requested (posterior p; metric log_inside a ... log_new b)": the margin a - b
(nats, metric log-evidence of the remembered room minus a new room; the appearance term is logged 0.00 "unavailable"), and the
proposals room_concept WITHDREW. These are the build's comparison; there is no eq.-roomF free energy in it.
Pose error after the decision: gt_error.csv's estimate (map frame) mapped to the world by ONE rigid fit over the start room's
first visit, median |est - truth| over [decision + 2 s, decision + 12 s]: a wrong recalled room pose shows up as an offset.
Objects restored (runs with LTOM): ltom "recall offered <id> (<class>) in room <name>" in the window, for the decided room.
Doors restored: door_concept "'<door>' RESTORED in remembered '<name>'" lines in the window (distinct doors).
"""
import csv, math, re, statistics, sys
from collections import defaultdict

import os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wbt_rooms
_R = wbt_rooms.rooms()[0]
# wbt names -> the graph's names in the fixture/tours (wbt room_3 = bedroom = graph room_2; wbt room_2 = bathroom = graph room_3)
_POLY = {'room_1': _R['room_1']['poly'], 'room_2': _R['room_3']['poly'], 'room_3': _R['room_2']['poly']}

def side_of(x, y):
    for k, poly in _POLY.items():
        if wbt_rooms.inside(poly, np.array([[x, y]]))[0]: return k
    return None   # in a doorway or a wall: no room

def load_gt(run):
    rows = []
    with open(f'{run}/gt_pose.tsv', newline='') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            try:
                rows.append({'t': int(r['t_ms']), 'x': float(r['x']), 'y': float(r['y']),
                             'n': {k: int(r[k]) for k in ('room_concept', 'ltsm_agent', 'door_concept', 'controller')}})
            except (KeyError, ValueError, TypeError):
                pass
    return rows

def line_time(gt, log, n):
    for r in gt:
        if r['n'][log] >= n: return r['t']
    return None

def grep(run, log, pat):
    out = []
    try:
        with open(f'{run}/logs/{log}.log', errors='replace') as f:
            for i, l in enumerate(f, 1):
                m = re.search(pat, l)
                if m: out.append((i, m))
    except FileNotFoundError:
        pass
    return out

def kabsch(src, dst):
    n = len(src); sx = sum(p[0] for p in src) / n; sy = sum(p[1] for p in src) / n
    dx = sum(p[0] for p in dst) / n; dy = sum(p[1] for p in dst) / n
    a = b = 0.0
    for (px, py), (qx, qy) in zip(src, dst):
        px -= sx; py -= sy; qx -= dx; qy -= dy
        a += px * qx + py * qy; b += px * qy - py * qx
    th = math.atan2(b, a); c, s = math.cos(th), math.sin(th)
    return lambda x, y: (c * (x - sx) - s * (y - sy) + dx, s * (x - sx) + c * (y - sy) + dy)

def grade(run):
    gt = load_gt(run)
    if len(gt) < 10: return None
    warm = __import__('os').path.exists(f'{run}/legs.tsv')
    # entries: truth room changes held >= 3 s
    entries = [(gt[0]['t'], side_of(gt[0]['x'], gt[0]['y']))]
    cand = None
    for r in gt:
        s = side_of(r['x'], r['y'])
        if s is None: continue
        if s == entries[-1][1]: cand = None; continue
        if cand is None or cand[1] != s: cand = (r['t'], s)
        elif r['t'] - cand[0] >= 3000: entries.append(cand); cand = None
    T = lambda log, n: line_time(gt, log, n)
    changes = [(T('controller', i), m.group(2)) for i, m in grep(run, 'controller', r'current room changed: (\S+) -> (\S+)')]
    accepted = [(T('room_concept', i), m.group(1)) for i, m in grep(run, 'room_concept', r'\[room\]\[proposal\] ACCEPTED: room \d+ is (\S+?),')]
    passage = [(T('room_concept', i), m.group(1), m.group(2)) for i, m in grep(run, 'room_concept', r'\[room\]\[recall\] (\S+) via (\S+): T_door_recall')]
    swaps = [(T('ltsm_agent', i), float(m.group(2)), float(m.group(3)), float(m.group(4)), m.group(1))
             for i, m in grep(run, 'ltsm_agent', r'\[recog\] SWAP requested: room_id (\d+) \(posterior ([-\d.]+); metric log_inside ([-\d.]+).*?log_new ([-\d.]+)')]
    withdrawn = [T('ltsm_agent', i) for i, m in grep(run, 'ltsm_agent', r'\[swap\] WITHDRAWN')]
    proposed = [T('ltsm_agent', i) for i, m in grep(run, 'ltsm_agent', r'\[swap\] PROPOSED')]
    verdicts = sorted(withdrawn + [T('ltsm_agent', i) for i, m in grep(run, 'ltsm_agent', r'\[swap\] ACCEPTED')])
    restored = [(T('door_concept', i), m.group(1), m.group(2)) for i, m in grep(run, 'door_concept', r"\[restore\] '(\S+)' RESTORED in remembered '(\S+)'")]
    # ltom (warm tours with LTOM): "<epoch s> [ltom] recall offered <id> (<class>) in room <name>", timed by its own clock
    offered = [(int(float(m.group(1)) * 1000), m.group(2), m.group(3)) for i, m in
               grep(run, 'ltom_agent', r'^(\d+\.\d+) \[ltom\] recall offered \S+ \((\w+)\) in room (\S+)')]
    has_ltom = __import__('os').path.exists(f'{run}/logs/ltom_agent.log')
    startup = [(T('room_concept', i), m.group(1)) for i, m in grep(run, 'room_concept', r"MapMode=ltsm: layout of '(\S+)'")]
    # map -> world: one rigid fit over the start room's first visit (from 10 s in)
    est = []
    try:
        with open(f'{run}/gt_error.csv', newline='') as f:
            for r in csv.DictReader(f):
                try: est.append((int(r['ts_ms']), float(r['est_x']), float(r['est_y']), float(r['gt_x_world']), float(r['gt_y_world'])))
                except (KeyError, ValueError): pass
    except FileNotFoundError:
        pass
    t_first_cross = entries[1][0] if len(entries) > 1 else gt[-1]['t']
    fit_rows = [e for e in est if gt[0]['t'] + 10000 <= e[0] <= t_first_cross - 2000][::5]
    to_world = kabsch([(e[1], e[2]) for e in fit_rows], [(e[3], e[4]) for e in fit_rows]) if len(fit_rows) >= 20 else None
    def pose_err(t):
        if to_world is None or t is None: return None
        d = [math.hypot(*(a - b for a, b in zip(to_world(e[1], e[2]), (e[3], e[4])))) for e in est if t + 2000 <= e[0] <= t + 12000]
        return statistics.median(d) if d else None
    name_of = {}   # truth room -> its remembered name
    if warm: name_of = {'room_1': 'room_1', 'room_2': 'room_2', 'room_3': 'room_3'}
    out = []
    for k, (tc, truth) in enumerate(entries):
        t_end = entries[k + 1][0] if k + 1 < len(entries) else gt[-1]['t'] + 1
        w0 = tc - 5000 if k else gt[0]['t'] - 120000
        inw = lambda t: t is not None and w0 <= t < t_end
        acc = [a for a in accepted if inw(a[0])]; pas = [p for p in passage if inw(p[0])]
        chg = [c for c in changes if inw(c[0])]
        if k == 0 and warm and startup:
            mech, t_dec, name = 'start', startup[0][0], startup[0][1]
        elif k == 0:
            mech, t_dec, name = 'new', gt[0]['t'], 'room_1'
        elif acc:
            mech, t_dec, name = 'recognition', acc[-1][0], acc[-1][1]
        elif pas:
            mech, t_dec, name = 'passage', pas[0][0], pas[0][1]
        elif chg:
            mech, t_dec, name = 'new', chg[0][0], chg[0][1]
        else:
            mech, t_dec, name = 'undecided', None, ''
        remembered = truth in name_of
        if not remembered and not warm and name and mech != 'undecided':
            others = set(name_of.values())
            verdict = 'false_recall' if name in others else ('ok_new' if mech == 'new' else 'wrong_mechanism')
            if name not in others: name_of[truth] = name
        elif remembered:
            verdict = 'ok_recall' if name == name_of[truth] else ('duplicate' if mech in ('new', 'undecided') else 'wrong_room')
        else:
            verdict = 'undecided'
        if verdict == 'duplicate' and any(inw(t) and t >= tc and not any(v is not None and v > t for v in verdicts) for t in proposed):
            verdict = 'cut_off'   # a recall proposal was still awaiting room_concept's verdict when the fleet stopped
        sw = [s for s in swaps if inw(s[0]) and (k == 0 or s[0] >= tc)]   # this room's episode: requests after the crossing
        last = [s for s in sw if t_dec is None or s[0] <= t_dec or mech == 'new']   # a new room: the request that was turned down / pending
        dr = sorted({d for t, d, r in restored if inw(t)})
        out.append({'run': run.rstrip('/').split('/')[-1], 'entry': k, 'truth': truth, 'first_visit': int(not remembered),
                    'mechanism': mech, 'name': name, 'expected': name_of.get(truth, '(new)'), 'verdict': verdict,
                    't_from_crossing_s': '' if t_dec is None or k == 0 else f'{(t_dec - tc) / 1000:.1f}',
                    'swap_requests': len(sw), 'withdrawn': len([t for t in withdrawn if inw(t)]),
                    'posterior': f'{last[-1][1]:.3f}' if last else '', 'margin_nats': f'{last[-1][2] - last[-1][3]:.2f}' if last else '',
                    'max_posterior': f'{max(s[1] for s in sw):.3f}' if sw else '',
                    'pose_err_m': '' if pose_err(t_dec) is None else f'{pose_err(t_dec):.3f}',
                    'doors_restored': ' '.join(dr),
                    'objects_offered': ('' if not has_ltom else ' '.join(sorted(c for t, c, rn in offered if inw(t) and rn == name)) or '0')})
    with open(f'{run}/c2_entries.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    return out

allrows = []
for run in sys.argv[1:]:
    r = grade(run)
    if r is None: print(f'{run}: no gt_pose.tsv'); continue
    allrows += r
    for e in r:
        print(f"{e['run']:10s} #{e['entry']:<2d} truth {e['truth']} {'first' if e['first_visit'] else 'again'} -> {e['mechanism']:11s} "
              f"{e['name']:8s} ({e['verdict']}) dt {e['t_from_crossing_s'] or '-':>5s} s  swaps {e['swap_requests']} wd {e['withdrawn']} "
              f"post {e['posterior'] or '-'} margin {e['margin_nats'] or '-'}  pose {e['pose_err_m'] or '-'} m  doors [{e['doors_restored']}]"
              + (f"  objects [{e['objects_offered']}]" if e['objects_offered'] else ''))
re_ = [e for e in allrows if not e['first_visit']]
fi = [e for e in allrows if e['first_visit'] and e['entry'] > 0]
print(f"\nre-entries (incl. warm start): {len(re_)}  correct recalls {sum(e['verdict'] == 'ok_recall' for e in re_)}  "
      f"duplicates {sum(e['verdict'] == 'duplicate' for e in re_)}  wrong room {sum(e['verdict'] == 'wrong_room' for e in re_)}  "
      f"cut off (recall pending at the stop) {sum(e['verdict'] == 'cut_off' for e in re_)}")
print(f"  by mechanism: " + ', '.join(f"{m} {sum(e['mechanism'] == m for e in re_)}" for m in ('start', 'passage', 'recognition', 'new', 'undecided')))
print(f"first entries into an unremembered room (negative control): {len(fi)}  new {sum(e['verdict'] == 'ok_new' for e in fi)}  "
      f"false recalls {sum(e['verdict'] == 'false_recall' for e in fi)}  with a swap requested {sum(e['swap_requests'] > 0 for e in fi)}")
m = [float(e['margin_nats']) for e in allrows if e['margin_nats'] and e['mechanism'] == 'recognition']
if m: print(f"recognition margins (log_inside - log_new, nats) at accepted recalls: {sorted(m)}")
mf = [float(e['max_posterior']) for e in fi if e['max_posterior']]
if mf: print(f"max recogniser posterior on first entries: {sorted(mf)}")
d = [float(e['t_from_crossing_s']) for e in re_ if e['t_from_crossing_s']]
if d: print(f"crossing -> decision (s): median {statistics.median(d):.1f}, range {min(d):.1f}..{max(d):.1f}")
p = [float(e['pose_err_m']) for e in re_ if e['pose_err_m']]
if p: print(f"pose error after recall (m): median {statistics.median(p):.3f}, max {max(p):.3f}")
