#!/usr/bin/env python3
"""A/B report (v2: policy-independent. leg_s = driver log "leg N (...): OK in S s" (--driver-log, default ab7_driver.log
section "##### <run>"), else policy_decisions fallback. cross_s/jamb_m = from gt_error.csv alone: contiguous span inside 1.5 m
of a wbt door centre whose side of the door line flips (|side|>0.3 m) = one crossing; cross_s = span duration; jamb_m as below
over that span. Crossings map 1:1 to finished legs, else n/a. Needs gt_error.csv, which is collected only at run end.)
 for DOOR_1 round-trip runs.   python3 ab_report.py <run_dir> [<run_dir> ...]

Per leg (rows) and per run (TOTAL rows), plus per-arm aggregates for runs named <prefix>_<ARM>* (ab7_A*, ab7_B*, ...).

SOURCES / METHODS
 outcome   controller.log "mission 'leg N': started|completed|failed at step K: <reason>|aborted" (no timestamps in that log).
 time      controller.log has NO wall clock, so leg time comes from policy_decisions.csv (t_ms, who). Each contiguous
           `cross ...` goal block is one leg's crossing and ends that leg (blocks are 1:1 with legs in rtD1i/tour3E; a
           retry that re-enters cross would add a block and the count check below turns times into n/a). leg_s = end(leg) - end(leg-1) (leg 1: first row);
           cross_s = first `cross` row of the leg -> leg end (retries included). Mapped to legs in order; n/a if the
           block count disagrees with the number of leg starts that finished.
 contacts  n/a unless a real contact source exists. Checked: controller.log / events.jsonl / crossing.log carry only the
           PREDICTED P(contact) of the look-ahead policy (not an event); shadow-bridge-dds neither logs nor reads Webots
           contacts/bumpers. A column is printed only as "n/a".
 stuck     controller.log lines, per leg segment (started -> next started): Stuck = "[policy] ... -> Stuck" transitions;
           hand = "handed back" lines; retry = "retry i/n after"; refused = "TARGET REFUSED"; fail = approach_timeout /
           door_wont_open / "not through" occurrences.
 jamb      Minimum clearance (m) between the robot and the two door_1 jamb points DURING THE CROSS WINDOW, using WEBOTS
           GROUND TRUTH (gt_error.csv gt_x_world/gt_y_world), not the estimate. Jambs from threerooms-piso.wbt DOOR_0 and
           DOOR_1 (centre, rotation, size.y=1.0 m, minus the 0.05 m frame => +-0.45 m); the door the window nears is used
           (mission door names differ per mission, e.g. tour3E crosses door_2). Robot = inscribed circle r=0.23 m
           (body is 0.46 wide x 0.65 long; heading is not in the world-frame CSV, so the circle is the optimistic bound
           for a straight crossing and ignores corner swing). Only samples within ZONE=1.2 m of the door centre count.
           clearance = min over samples/jambs of (dist(robot, jamb) - 0.23). n/a if the window never comes within ZONE of a known door.
"""
import sys, os, re, csv, glob, math
import numpy as np

DEFAULT_DRV = '/tmp/claude-1000/-home-pbustos-robocomp-components-active-inference-room-concept/d48d7b74-1a76-458d-aea2-e6fc124d5fa8/scratchpad/ab7_driver.log'
R_BODY = 0.23
ZONE = 1.2
HALF = 0.5 - 0.05
def _door(c, rot):
    c = np.array(c); ax = np.array([math.sin(-rot), math.cos(-rot)])     # local +y rotated by rot about z
    return c, np.array([c + HALF * ax, c - HALF * ax])
DOORS = {'DOOR_0': _door((-3.59333, 7.22555), -1.57), 'DOOR_1': _door((-4.76901, 9.3796), -1.5857206)}

def parse_log(path):
    """-> dict leg -> dict(status, reason, counts)"""
    legs, cur = {}, None
    if not os.path.exists(path): return legs
    for l in open(path, errors='replace'):
        m = re.search(r"mission 'leg (\d+)': (\w+)(?: at step \d+: (\w+))?", l)
        if m:
            n = int(m.group(1)); st = m.group(2)
            if st == 'started':
                cur = legs.setdefault(n, dict(status='running', reason='', Stuck=0, hand=0, retry=0, refused=0, fail=0))
            elif n in legs:
                legs[n]['status'] = st; legs[n]['reason'] = m.group(3) or ''
            continue
        if cur is None: continue
        if re.search(r"\[policy\].* -> Stuck", l): cur['Stuck'] += 1
        if 'handed back' in l: cur['hand'] += 1
        if re.search(r"retry \d+/\d+ after", l): cur['retry'] += 1
        if 'TARGET REFUSED' in l: cur['refused'] += 1
        cur['fail'] += len(re.findall(r"approach_timeout|door_wont_open|not through", l))
    return legs

def cross_blocks(path):
    """-> list of (t_start, t_end, followed_by_explore) for contiguous `cross` goal blocks"""
    if not os.path.exists(path): return [], 0
    rows = [(int(r['t_ms']), r['who']) for r in csv.DictReader(open(path))]
    blocks, i = [], 0
    while i < len(rows):
        if rows[i][1].startswith('cross'):
            j = i
            while j + 1 < len(rows) and rows[j + 1][1].startswith('cross'): j += 1
            nxt = rows[j + 1][1] if j + 1 < len(rows) else None
            blocks.append((rows[i][0], rows[j][0], nxt is None or nxt == 'afford_room'))
            i = j + 1
        else: i += 1
    return blocks, rows[0][0] if rows else 0

def leg_windows(path):
    blocks, t0 = cross_blocks(path); legs, first, prev_end = [], None, t0
    for s, e, final in blocks:
        first = s if first is None else first
        legs.append((prev_end, first, e)); prev_end, first = e, None
    return legs   # (leg_start, first_cross, leg_end)

def gt_samples(path):
    if not os.path.exists(path): return None
    with open(path) as f:
        h = next(csv.reader(f)); it, ix, iy = h.index('ts_ms'), h.index('gt_x_world'), h.index('gt_y_world')
        a = np.array([[float(r[it]), float(r[ix]), float(r[iy])] for r in csv.reader(f) if len(r) > iy])
    return a

def jamb_clear(gt, t0, t1):
    """min over wbt doors with samples inside ZONE of (dist(robot, jamb) - R_BODY); n/a if the window never nears a door"""
    if gt is None: return None
    m = (gt[:, 0] >= t0) & (gt[:, 0] <= t1)
    best = None
    for c, jambs in DOORS.values():
        p = gt[m, 1:3]; p = p[np.linalg.norm(p - c, axis=1) <= ZONE]
        if len(p):
            d = float(np.min([np.linalg.norm(p - j, axis=1) for j in jambs]) - R_BODY)
            best = d if best is None else min(best, d)
    return best


ZONE_X = 1.5
def gt_crossings(gt):
    """Policy-independent passages from GT alone: per wbt door, contiguous spans inside ZONE_X of its centre where the
    signed side of the door line flips between span start and end. -> sorted [(t_in, t_out, jamb_clear_m)]"""
    out = []
    if gt is None: return out
    for c, jambs in DOORS.values():
        ax = (jambs[0] - jambs[1]) / np.linalg.norm(jambs[0] - jambs[1]); nrm = np.array([-ax[1], ax[0]])
        d = np.linalg.norm(gt[:, 1:3] - c, axis=1); inz = d <= ZONE_X; i = 0
        while i < len(gt):
            if not inz[i]: i += 1; continue
            j = i
            while j + 1 < len(gt) and inz[j + 1]: j += 1
            q = gt[i:j + 1, 1:3] - c; sd = q @ nrm; lat = q @ ax
            idx = np.where(np.abs(sd) > 0.3)[0]
            # a crossing = the side (|sd|>0.3 deadband) flips AND the line is crossed through the opening (|lateral| <= HALF)
            flips = [(a, b) for a, b in zip(idx, idx[1:]) if sd[a] * sd[b] < 0 and
                     abs(np.interp(0, [sd[a], sd[b]][::(1 if sd[a] < sd[b] else -1)], [lat[a], lat[b]][::(1 if sd[a] < sd[b] else -1)])) <= HALF]
            for a, b in flips:
                p = gt[i:j + 1, 1:3]
                cl = float(np.min([np.linalg.norm(p - q_, axis=1) for q_ in jambs]) - R_BODY)
                out.append((gt[i, 0], gt[j, 0], cl))
            i = j + 1
    return sorted(out)

def driver_legs(path, run):
    """-> {leg: (ok, seconds)} from the round-trip driver log section '##### <run> ...'"""
    res, on = {}, False
    if not path or not os.path.exists(path): return res
    for l in open(path, errors='replace'):
        if l.startswith('#####'): on = (l.split()[1] == run); continue
        m = re.match(r"leg (\d+) .*?: (OK in (\d+(?:\.\d+)?) s|FAILED.*)", l)
        if on and m: res[int(m.group(1))] = (m.group(3) is not None, float(m.group(3)) if m.group(3) else None)
    return res

def fmt(v, f='{:.1f}'): return 'n/a' if v is None else f.format(v)

def analyse(run, driver_log=None):
    name = os.path.basename(run)
    log = parse_log(os.path.join(run, 'logs', 'controller.log'))
    drv = driver_legs(driver_log, name)
    for n in drv: log.setdefault(n, dict(status='completed' if drv[n][0] else 'failed', reason='', Stuck=0, hand=0, retry=0, refused=0, fail=0))
    win = leg_windows(os.path.join(run, 'policy_decisions.csv'))   # cross-check / fallback only
    gt = gt_samples(os.path.join(run, 'gt_error.csv')); cr = gt_crossings(gt)
    legs = sorted(log); done = [n for n in legs if log[n]['status'] != 'running']
    gt_ok = len(cr) == len(done) and len(cr) > 0
    pol_ok = len(win) == len(done) and len(win) > 0
    note = []
    if gt is None: note.append('gt_error.csv not collected yet: cross_s/jamb n/a')
    elif not gt_ok: note.append(f'{len(cr)} GT crossings vs {len(done)} finished legs: cross_s/jamb n/a')
    if not drv and not pol_ok: note.append('no driver log / policy rows: leg_s n/a')
    if gt_ok and pol_ok and drv == {} : pass
    out = []
    for k, n in enumerate(legs):
        d = log[n]; fin = n in done
        leg_s = drv[n][1] if n in drv else ((win[k][2] - win[k][0]) / 1e3 if pol_ok and fin else None)
        cross_s = jc = None
        if gt_ok and fin: cross_s = (cr[k][1] - cr[k][0]) / 1e3; jc = cr[k][2]
        out.append(dict(leg=n, status=d['status'] + (':' + d['reason'] if d['reason'] else ''), leg_s=leg_s, cross_s=cross_s,
                        contact=None, Stuck=d['Stuck'], hand=d['hand'], retry=d['retry'], refused=d['refused'], fail=d['fail'], jamb=jc))
    return out, '; '.join(note), (len(cr), len(win), len(done))

HDR = f"{'run':<10}{'leg':>4} {'status':<24}{'leg_s':>7}{'cross_s':>8}{'contact':>8}{'stuck':>6}{'hand':>5}{'retry':>6}{'refus':>6}{'fail':>5}{'jamb_m':>8}"
def row(name, leg, r):
    return (f"{name:<10}{leg:>4} {r['status']:<24}{fmt(r['leg_s']):>7}{fmt(r['cross_s']):>8}{'n/a':>8}{r['Stuck']:>6}{r['hand']:>5}"
            f"{r['retry']:>6}{r['refused']:>6}{r['fail']:>5}{fmt(r['jamb'], '{:.3f}'):>8}")

def summary(name, rs, label):
    ok = sum(r['status'] == 'completed' for r in rs)
    med = lambda k: (float(np.median([r[k] for r in rs if r[k] is not None])) if any(r[k] is not None for r in rs) else None)
    jm = [r['jamb'] for r in rs if r['jamb'] is not None]
    t = {k: sum(r[k] for r in rs) for k in ('Stuck', 'hand', 'retry', 'refused', 'fail')}
    return (f"{name:<10}{label:>4} {str(ok) + '/' + str(len(rs)) + ' done':<24}{fmt(med('leg_s')):>7}{fmt(med('cross_s')):>8}{'n/a':>8}"
            f"{t['Stuck']:>6}{t['hand']:>5}{t['retry']:>6}{t['refused']:>6}{t['fail']:>5}{fmt(min(jm) if jm else None, '{:.3f}'):>8}")

def main():
    global DRV
    args = sys.argv[1:]; DRV = None
    if '--driver-log' in args:
        i = args.index('--driver-log'); DRV = args[i + 1]; del args[i:i + 2]
    elif os.path.exists(DEFAULT_DRV): DRV = DEFAULT_DRV
    runs = [r.rstrip('/') for r in args]
    if not runs: print(__doc__); return
    print(HDR); arms = {}
    for run in runs:
        name = os.path.basename(run); rs, note, cnt = analyse(run, DRV)
        for r in rs: print(row(name, r['leg'], r))
        print(summary(name, rs, 'ALL') + (f"   [{note}]" if note else '') + f"   (GT crossings {cnt[0]}, policy blocks {cnt[1]}, finished legs {cnt[2]})")
        m = re.match(r'(ab\d+_[A-Za-z])', name)
        if m: arms.setdefault(m.group(1), []).extend(rs)
    if arms:
        print('\nPER-ARM (median leg_s/cross_s over legs; counts summed; jamb_m = min)')
        for a, rs in sorted(arms.items()): print(summary(a, rs, 'ARM'))
    print("\ncontact: n/a (no contact/bumper source exists: bridge logs none; controller logs only predicted P(contact)). "
          "jamb_m: Webots GT, inscribed circle r=0.23, zone 1.2 m, see docstring.")

if __name__ == '__main__': main()
