#!/usr/bin/env python3
"""AuRo C3(b) change test: one run folder -> timeline CSV + summary.  c3b_timeline.py <run_dir>

Inputs (all written by the tour, c3b series 2026-11-21):
  chair_ai2_log.csv  chair_concept's per-cycle belief: node, cx, cy (room frame), std_cx, std_cy, nbv_gain_pub, t_ms
  chair_move.txt     "chair moved t_ms <ms>" and the commanded/read world pose of CHAIR_3
  legs.tsv           per-leg t0/t1: leaving room_2 = leg 2 t0, back in room_2 = leg 3 t1 (the RETURN)
  graph_trace.csv    `executing` edges: which offer the controller claimed, and when
Output: <run_dir>/c3b_timeline.csv and a summary on stdout.

Which belief is the moved chair: the node with the most belief rows while the robot was in room_2 before leaving (the
look round there). Corrected: the first time after the return that ANY chair belief lies 1.0 +- 0.25 m from that node's
pre-move position (a rotation-free test: the room frame is not the world frame, the displacement norm is the same in
both) with std_cx and std_cy below 0.15 m. Reported both for the same node (corrected in place) and any node (a new
instance born at the new spot).
"""
import csv, math, statistics, sys
from collections import defaultdict

run = sys.argv[1].rstrip('/')
MOVE_M, TOL_M, STD_MAX = 1.0, 0.25, 0.15

rows = []
with open(f'{run}/chair_ai2_log.csv', newline='') as f:
    for r in csv.DictReader(f):
        try:
            rows.append({'t': int(r['t_ms']), 'node': r['node'], 'cx': float(r['cx']), 'cy': float(r['cy']),
                         'sx': float(r['std_cx']), 'sy': float(r['std_cy']), 'gain': float(r['nbv_gain_pub'])})
        except (KeyError, ValueError):
            continue
legs = {}
with open(f'{run}/legs.tsv', newline='') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        legs[int(r['leg'])] = (int(r['t0_ms']), int(r['t1_ms']), r['status'])
t_move = None
for l in open(f'{run}/chair_move.txt'):
    if l.startswith('chair moved t_ms'):
        t_move = int(l.split()[-1])
room_of = {}   # chair node name -> the room it was hung under (RT edge room -> chair), from the trace
claims = []   # (t, 'on'|'off', affordance)
with open(f'{run}/graph_trace.csv', newline='') as f:
    for r in csv.DictReader(f):
        if r['edge_type'] == 'RT' and r['event'] == 'edge_insert' and r['b_name'].startswith('chair_'):
            room_of[r['b_name']] = r['a_name']
        if r['edge_type'] == 'executing' and r['event'] in ('edge_insert', 'edge_delete'):
            claims.append((int(r['t_ms']), 'on' if r['event'] == 'edge_insert' else 'off', r['b_name']))
# object offers are claimed without an `executing` edge in the trace; chair_concept logs each claim with a wall clock
import os, re
if os.path.exists(f'{run}/logs/chair_concept.log'):
    for l in open(f'{run}/logs/chair_concept.log', errors='replace'):
        m = re.match(r"^(\d+\.\d+) \[affordance\] '(chair_\d+)' \u2192 (\w+)", l)
        if m:
            claims.append((int(float(m.group(1)) * 1000), 'on' if m.group(3) == 'executing' else 'off', 'aff_' + m.group(2)))
claims.sort()

def claimed_at(t):
    live = {}
    for tc, ev, aff in claims:
        if tc > t: break
        if ev == 'on': live[aff] = tc
        else: live.pop(aff, None)
    return '|'.join(sorted(live)) if live else ''

if 1 not in legs or 2 not in legs or t_move is None:
    sys.exit(f'{run}: incomplete (legs {sorted(legs)}, move {t_move})')
in_r2 = (legs[1][1], legs[2][0])                       # in room_2 before leaving: leg 1 end .. leg 2 start
t_ret = legs[3][1] if 3 in legs and legs[3][2] == 'OK' else None
count = defaultdict(int)
for r in rows:
    if in_r2[0] <= r['t'] <= in_r2[1]: count[r['node']] += 1
if not count:
    sys.exit(f'{run}: no chair belief while in room_2 before leaving')
target = max(count, key=count.get)
pre = [r for r in rows if r['node'] == target and in_r2[0] <= r['t'] <= in_r2[1]]
tail = pre[max(0, len(pre) - 50):]
px, py = statistics.median(r['cx'] for r in tail), statistics.median(r['cy'] for r in tail)
psx, psy = tail[-1]['sx'], tail[-1]['sy']

def phase(t):
    if t < in_r2[0]: return 'before_room2'
    if t <= in_r2[1]: return 'learning_room2'
    if t < t_move: return 'leaving'
    if t_ret is None or t < t_ret: return 'away'
    return 'after_return'

corr_same = corr_any = None
with open(f'{run}/c3b_timeline.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['t_ms', 't_from_return_s', 'phase', 'node', 'is_moved_chair', 'cx', 'cy', 'std_cx', 'std_cy',
                'displacement_m', 'offer_price_nats', 'controller_claim'])
    for r in rows:
        d = math.hypot(r['cx'] - px, r['cy'] - py)
        ph = phase(r['t'])
        w.writerow([r['t'], '' if t_ret is None else f"{(r['t'] - t_ret) / 1000:.2f}", ph, r['node'], int(r['node'] == target),
                    f"{r['cx']:.4f}", f"{r['cy']:.4f}", f"{r['sx']:.4f}", f"{r['sy']:.4f}", f'{d:.3f}', f"{r['gain']:.4f}",
                    claimed_at(r['t'])])
        ok = ph == 'after_return' and room_of.get(r['node']) == room_of.get(target) and abs(d - MOVE_M) <= TOL_M and r['sx'] < STD_MAX and r['sy'] < STD_MAX
        if ok and corr_any is None: corr_any = r
        if ok and r['node'] == target and corr_same is None: corr_same = r

def last_chair_claim_before(t):
    c = [(tc, aff) for tc, ev, aff in claims if ev == 'on' and tc <= t and 'chair' in aff]
    return c[-1] if c else None

print(f'run {run}')
print(f'  moved chair = {target}: pre-move belief ({px:.3f},{py:.3f}) room frame, std ({psx:.3f},{psy:.3f}) m')
print(f'  chair moved at t={t_move}; back in room_2 at t={t_ret}')
after = [(tc, aff) for tc, ev, aff in claims if ev == 'on' and t_ret is not None and tc >= t_ret]
print(f'  offers claimed after return: {", ".join(a for _, a in after[:12]) or "none"}')
for name, c in (('same node (corrected in place)', corr_same), ('any chair belief', corr_any)):
    if c is None:
        print(f'  corrected [{name}]: NO')
        continue
    lc = last_chair_claim_before(c['t'])
    by = f"{lc[1]} claimed {(c['t'] - lc[0]) / 1000:.1f} s before" if lc and t_ret and lc[0] >= t_ret else 'no chair offer claimed since the return'
    print(f"  corrected [{name}]: {c['node']} at {(c['t'] - t_ret) / 1000:.1f} s after return, displacement "
          f"{math.hypot(c['cx'] - px, c['cy'] - py):.2f} m, std ({c['sx']:.3f},{c['sy']:.3f}); claim then: "
          f"'{claimed_at(c['t'])}'; {by}")
print(f'  timeline: {run}/c3b_timeline.csv')
