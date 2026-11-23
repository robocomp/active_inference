#!/usr/bin/env python3
"""AuRo item 14 (2026-11-22): removal test report.   c14_report.py <run_dir>... [--csv out.csv]

The tour (scratchpad c14_removal_1122.sh) moves TABLE_ROUND + CHAIR_2 (room_1) and NIGHTSTAND_R (room_2) out of the house after
leg 2; the robot returns to room_2 (leg 3) and room_1 (leg 4). Homes come from crossing/c14_series_1122.txt ("homes:").
Per removed object:
  NODES        the concept agent's beliefs whose pre-removal position (world frame, run_frames.py) is within MATCH_M of home,
               and their LTOM uids ("[ltom] born <node> uid=<uid>").
  IN VIEW      seconds after the return to its room that the home spot is in view (c3b_report.py's 4 m / 110 deg / walls rule).
  LIVE AFTER   any belief of that class within MATCH_M of home after the return (a re-birth there would be a phantom), and
               how it ended (graph node_delete time).
  RECALL       ltom_recall_<uid> offered in its room after the return: when, and whether it was withdrawn before the robot left.
  MEMORY END   the LTOM entry at the end of the run (ltom_repertory_end.json): present?, exist_prob, number of lives, last end.
Per run, FALSE REMOVALS: LTOM uids of objects NOT removed that are missing from the end repertory, and every
"[existence] REMOVE" an agent logged while in a room, with its time.
"""
import csv, json, math, os, re, statistics, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_frames import make_to_world
import c3b_report as V          # visibility rule: V.blocked, V.RANGE_M, V.HFOV_DEG

MATCH_M = 0.5      # a belief AT the home spot after the return
PICK_M = 1.0       # pre-removal: the NEAREST belief of the class within this of home is the object's (c14_r1: CHAIR_2's 81-row belief was 0.55 m off)
OBJ = {'TABLE_ROUND': ('table', 'room_1', 4), 'CHAIR_2': ('chair', 'room_1', 4), 'NIGHTSTAND_R': ('nightstand', 'room_2', 3)}


def homes(run):
    s = open(os.path.join(os.path.dirname(run), 'c14_series_1122.txt')).read()
    line = [l for l in s.splitlines() if 'homes:' in l][-1]
    return {k: tuple(map(float, v.split(','))) for k, v in re.findall(r'(\w+)=([-\d.]+,[-\d.]+)', line.split('homes:')[1])}


def rows_of(run, cls):
    p = f'{run}/{cls}_ai2_log.csv'
    if not os.path.exists(p): return []
    out = []
    for r in csv.DictReader(open(p)):
        try:
            t = int(r.get('t_ms') or r['pkt_ts'])
            out.append((t, r['node'], float(r['cx']), float(r['cy']), max(float(r['std_cx']), float(r['std_cy']))))
        except (KeyError, ValueError, TypeError): pass
    return out


def in_view_s(gt, t0, t1, p):
    tot = 0.0; first = None
    for a, b in zip(gt, gt[1:]):
        if b[0] < t0 or a[0] > t1: continue
        n = max(1, int((b[0] - a[0]) / 100)); dyaw = math.remainder(b[3] - a[3], 2 * math.pi)
        for k in range(n):
            f = k / n; t = a[0] + f * (b[0] - a[0]); dt = (b[0] - a[0]) / n / 1000
            if t < t0 or t > t1 or dt > 2.0: continue
            x, y, yaw = a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2]), a[3] + f * dyaw
            if math.hypot(p[0] - x, p[1] - y) > V.RANGE_M: continue
            if abs(math.remainder(math.atan2(p[1] - y, p[0] - x) - yaw - math.pi / 2, 2 * math.pi)) > math.radians(V.HFOV_DEG / 2): continue
            if V.blocked((x, y), p): continue
            tot += dt; first = first if first is not None else t
    return tot, first


def one(run):
    run = run.rstrip('/'); name = os.path.basename(run)
    H = homes(run)
    legs = {int(r['leg']): r for r in csv.DictReader(open(f'{run}/legs.tsv'), delimiter='\t')}
    t_rm = int(re.search(r'removed t_ms (\d+)', open(f'{run}/removed.txt').read()).group(1))
    gt = [(int(r['t_ms']), float(r['x']), float(r['y']), float(r['yaw'])) for r in csv.DictReader(open(f'{run}/gt_pose.tsv'), delimiter='\t')]
    t_end = gt[-1][0]
    tw = make_to_world(run)
    ltom = open(f'{run}/logs/ltom_agent.log', errors='replace').read().splitlines()
    born = [(float(m.group(1)) * 1000, m.group(2), m.group(3)) for l in ltom for m in [re.match(r'^(\d+\.\d+) \[ltom\] born (\S+) uid=(\d+)', l)] if m]
    rec = [(float(m.group(1)) * 1000, m.group(2), m.group(3)) for l in ltom for m in [re.match(r'^(\d+\.\d+) \[ltom\] recall (offered|withdrawn) ltom_recall_(\d+)', l)] if m]
    deletes = {}
    for r in csv.DictReader(open(f'{run}/graph_trace.csv')):
        if r['event'] == 'node_delete': deletes.setdefault(r['a_name'], []).append(int(r['t_ms']))
    end = {e['uid']: e for e in json.load(open(f'{run}/ltom_repertory_end.json'))['entries']} if os.path.exists(f'{run}/ltom_repertory_end.json') else {}
    out = []
    removed_uids = set()
    for obj, (cls, room, leg) in OBJ.items():
        home = H[obj]
        rows = rows_of(run, cls)
        pre = {}
        for t, n, x, y, s in rows:
            if t < t_rm:
                w = tw(x, y, t)
                if w: pre.setdefault(n, []).append(w)
        dist = {n: math.hypot(statistics.median(w[0] for w in ws[-50:]) - home[0], statistics.median(w[1] for w in ws[-50:]) - home[1]) for n, ws in pre.items()}
        best = min(dist, key=dist.get) if dist else None
        nodes = [best] if best and dist[best] <= PICK_M else []
        uids = sorted({u for t, n, u in born if n in nodes and t < t_rm})
        removed_uids |= set(uids)
        ok = leg in legs and legs[leg]['status'] == 'OK'
        t_ret = int(legs[leg]['t1_ms']) if ok else None
        t_leave = int(legs[leg + 1]['t0_ms']) if ok and leg + 1 in legs else t_end
        d = dict(run=name, object=obj, cls=cls, room=room, nodes='|'.join(nodes), pre_err_m=round(dist[best], 2) if nodes else '', uids='|'.join(uids), returned=int(ok))
        if ok:
            v, f = in_view_s(gt, t_ret, t_leave, home)
            d.update(visit_s=round((t_leave - t_ret) / 1000, 1), home_in_view_s=round(v, 1), first_in_view_s='' if f is None else round((f - t_ret) / 1000, 1))
            live = [(t, n) for t, n, x, y, s in rows if t_ret <= t <= t_leave and (w := tw(x, y, t)) and math.hypot(w[0] - home[0], w[1] - home[1]) <= MATCH_M]
            d['live_belief_at_home_after_return'] = f"{len(live)} rows, nodes {'|'.join(sorted({n for _, n in live}))}" if live else 'none'
            offered = [t for t, ev, u in rec if u in uids and ev == 'offered' and t_ret - 15000 <= t <= t_leave]
            withdrawn = [t for t, ev, u in rec if u in uids and ev == 'withdrawn' and offered and t >= offered[0]]
            d['recall_offered_s'] = round((offered[0] - t_ret) / 1000, 1) if offered else 'no'
            d['recall_withdrawn_s'] = round((withdrawn[0] - t_ret) / 1000, 1) if withdrawn else ('' if not offered else 'held')
            d['recall_withdrawn_before_leaving'] = int(bool(withdrawn) and withdrawn[0] < t_leave - 3000)
        ents = [end[u] for u in uids if u in end]
        d['memory_at_end'] = 'absent' if not ents else '; '.join(
            f"{e['name']} p={e.get('exist_prob', float('nan')):.2f} lives={len(e.get('lives', []))} last_end={e.get('lives', [{}])[-1].get('end', '')}" for e in ents)
        out.append(d)
    # FALSE REMOVALS: memories of objects that were not removed, lost by the end; and every existence REMOVE line
    others = sorted({u for t, n, u in born if u not in removed_uids})
    lost = [u for u in others if end and u not in end]
    rm_lines = []
    for a in ('chair', 'table', 'nightstand'):
        p = f'{run}/logs/{a}_concept.log'
        if os.path.exists(p):
            for l in open(p, errors='replace'):
                m = re.match(r'^(\d+\.\d+) .*\[existence\] REMOVE id=(\d+) \((.*)\)', l)
                if m: rm_lines.append(f"{a}@{(float(m.group(1)) * 1000 - t_rm) / 1000:+.0f}s")
    out.append(dict(run=name, object='(run)', cls='', room='', nodes='', uids=f'{len(others)} other memories',
                    memory_at_end=f"lost {len(lost)}: {'|'.join(lost)}" if lost else 'none lost',
                    existence_removes='; '.join(rm_lines) or 'none'))
    return out


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    csv_out = sys.argv[sys.argv.index('--csv') + 1] if '--csv' in sys.argv else None
    if csv_out: args = [a for a in args if a != csv_out]
    rows = []
    for r in args:
        try: rows += one(r)
        except Exception as e: print(f'{r}: {type(e).__name__}: {e}')
    for d in rows: print(' | '.join(f'{k}={v}' for k, v in d.items()))
    if csv_out and rows:
        keys = list(dict.fromkeys(k for d in rows for k in d))
        with open(csv_out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
