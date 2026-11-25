#!/usr/bin/env python3
"""LIVE A/B: old countable engine (main) vs the shared engine, on the same scripted A/B/A visit of the same object in the same
Webots world (owner choice 2026-11-25: the replay dumps cannot feed the old engine the same evidence, task-SBS-report.md).
Not bit-identical inputs: the same scenario, two live runs.
    sbs_compare.py <run_dir> [<run_dir> ...]        (run dirs of parked_test.sh; one object per run)
Per run, metrics fixed before any old-engine result was seen (task-SBS-brief.md step 2, live form):
  node     the class node nearest the truth by aba_runlog.match_node_to_truth (d2 printed even when > chi2_2(99 %))
  final    error vs truth per DOF at the end of the A2 dwell (published room frame of the visit; bed h is an interval) and
           e / sigma per DOF (sigma = the engine's own logged std)
  NEES     mean over the node's updates inside the three dwells of sum_k (e_k / sigma_k)^2, per DOF count (6): ~1 when honest
  events   DEGENERATE re-seeds in the class agent's log; number of distinct nodes of the class
"""
import math, os, re, sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aba_runlog as R
from aba_geometry import wrap

DOFS = ('cx', 'cy', 'H', 'w', 'h', 'yaw')


def errors(est, T, fr, t):
    px, py, pyaw = fr.world_to_pub(T['cx'], T['cy'], T['yaw_theta_world'], t)
    truth = [px, py, T['H'], T['w'], None, pyaw]
    e = []
    for k in range(6):
        x = est[k]
        if k == 4: e.append(0.0 if T['h'][0] <= x <= T['h'][1] else min(x - T['h'][0], x - T['h'][1], key=abs))
        elif k == 5: e.append(wrap(x - truth[5]))
        else: e.append(x - truth[k])
    return e


def one(run):
    run = Path(run)
    S = R.segments(run)
    for defn, segs in S.items():
        plan = R.plan_of(run, defn)
        if plan is None or any(s not in segs for s in R.SEGS):
            print(f'{run.name} {defn}: incomplete visit'); continue
        T = plan['truth']; cls = T['cls']
        rows = R.ai2_rows(run, cls)
        m = R.match_node_to_truth(run, rows, segs, T)
        fr = m['frames']
        if fr is None: print(f'{run.name} {defn}: {m["why"]}'); continue
        node = m['node'] or (m.get('why', '').split()[2] if m.get('why', '').startswith('nearest node') else None)
        why = R.visit_validity(segs, R.controller_goal_tolerance()[0])
        print(f'== {run.name} {defn} ({cls}): {R.describe_match(m)}' + (f'  [visit: {"; ".join(why)}]' if why else ''))
        if node is None: continue
        ups = R.updates(rows, node)
        t_end = segs['A2']['t_dwell1_ms']
        d = R.last_before(ups, t_end)
        e = errors(d['est'], T, fr, t_end)
        print('   final ' + ' '.join(f'{DOFS[k]} {e[k]:+.4f} ({e[k] / d["sig"][k]:+.2f}s)' for k in range(6)))
        dw = [(segs[s]['t_dwell0_ms'], segs[s]['t_dwell1_ms']) for s in R.SEGS]
        q = [sum((ek / sk) ** 2 for ek, sk in zip(errors(u['est'], T, fr, u['t']), u['sig'])) / 6
             for u in ups if any(a <= u['t'] <= b for a, b in dw)]
        print(f'   NEES/DOF over the dwells: {sum(q) / len(q):.2f} (n={len(q)}, honest ~1)' if q else '   NEES: no dwell updates')
        logs = list((run / 'logs').glob(f'{cls}_concept*')) if (run / 'logs').exists() else []
        deg = sum(len(re.findall('DEGENERATE', p.read_text(errors='replace'))) for p in logs)
        print(f'   events: DEGENERATE {deg} ({len(logs)} log file(s)); {len({u["node"] for u in rows})} {cls} node(s)')


if __name__ == '__main__':
    for a in sys.argv[1:]: one(a)
