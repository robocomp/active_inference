#!/usr/bin/env python3
"""Room-recognition gold tour report (spec 2026-10-22).  python3 recognition_report.py <run_dir>

Prints PASS / FAIL / NOT MEASURED per criterion C1..C8; exits 1 on any FAIL. NOT MEASURED (an input the criterion needs
is absent) never counts as a pass and does not by itself fail the exit code.  Tour legs (recognition_tour.sh): 1 first entry of a new
room, 2 back, 3 first entry of a second new room, 4 the RECOGNITION leg (new door into a remembered room), 5 back through
the carried passage, 6 on to room_1 (ordinary recalls).

Inputs in <run_dir>:
  legs.tsv                 driver: leg, status, t0_ms, t1_ms (epoch), and the 0-based line range [x0,x1) each log held
                           during the leg (ltsm0/1, door0/1, room0/1); the agent logs carry no wall clock.
  logs/ltsm_agent.log, logs/door_concept.log, logs/room_concept.log
  ltsm_memory.json         copy of the ltsm DSR persistence at the end of the tour
  gt_error.csv             ts_ms, gt_x, gt_y, est_x, est_y and cov_xx (the localiser's xy variance, m^2, isotropic)
  recognition_trace.csv    ltsm's per-step trace (stamp_ms, provisional_id, act, room_id, 'id:post:log_inside:loglr;..;new:post::log_new'),
                           copied by the driver; C1/C2 read it with the leg t0_ms/t1_ms windows.  Absent => C1 MAP part NOT MEASURED.
Log text parsed (ltsm room_recognizer.cpp / room_swap.cpp / room_recall.cpp, door_concept specificworker.cpp):
  "[recog] SWAP requested: room_id N (posterior p)", "[swap] STAGED: P is <mem> (case ..", "[recog] UNDO requested",
  "[swap] UNDONE", "[recall] LATE ..", "[recall] recall_room_N posted", "[restore] 'X' recalled (late)",
  "[restore] provisional 'X' dropped", "[restore] 'X' entered FRESH", "[room][late-recall] X adopted".
"""
import csv, json, math, os, re, sys

P, F, NM = 'PASS', 'FAIL', 'NOT MEASURED'

CHI2_2_95 = 5.99        # chi-square, 2 dof, 0.95 quantile
FIRST_LEGS, RECOG_LEG, RETURN_LEGS = (1, 3), 4, (5, 6)
SWAP_PAT = re.compile(r'\[recog\] SWAP requested|\[swap\] STAGED|\[recall\] LATE|\[room\]\[late-recall\] .*adopted')

def read_lines(path):
    try:
        return open(path, errors='replace').read().splitlines()
    except OSError:
        return []

def read_legs(run):
    legs = {}
    try:
        for r in csv.DictReader(open(os.path.join(run, 'legs.tsv')), delimiter='\t'):
            legs[int(r['leg'])] = {k: (v if k == 'status' else int(v)) for k, v in r.items() if k != 'leg'}
    except OSError:
        pass
    return legs

def slice_of(lines, legs, leg, key):
    r = legs.get(leg)
    return lines[r[key + '0']:r[key + '1']] if r else []

def strip_id(name):
    return re.sub(r'^r\d+_', '', name)

def memory_rooms(run):
    """-> [(name, room_id or None, subtype)] of the non-staged room nodes, and the symbol table"""
    try:
        sym = json.load(open(os.path.join(run, 'ltsm_memory.json')))['DSRModel']['symbols']
    except (OSError, ValueError, KeyError):
        return None, {}
    out = []
    for n in sym.values():
        if n.get('type') != 'room': continue
        sub = (n.get('attribute', {}).get('object_subtype') or {}).get('value', '')
        if sub == 'staged': continue
        m = re.match(r'^r(\d+)_', n['name'])
        out.append((n['name'], int(m.group(1)) if m else None, sub))
    return out, sym

def trace_rows(run):
    """-> [(stamp_ms, act, room_id, p_new, {id: post})] or None when the trace file is absent"""
    p = os.path.join(run, 'recognition_trace.csv')
    if not os.path.exists(p): return None
    out = []
    for r in list(csv.reader(open(p)))[1:]:
        if len(r) < 5: continue
        cand, pn = {}, None
        for item in r[4].split(';'):
            f = item.split(':')
            if len(f) >= 2 and f[0] == 'new': pn = float(f[1])
            elif len(f) >= 2 and f[0]: cand[int(f[0])] = float(f[1])
        if pn is not None: out.append((int(r[0]), r[2], int(r[3] or 0), pn, cand))
    return out

def c1(legs, ltsm, run):
    bad = []
    for leg in FIRST_LEGS:
        if leg not in legs: return F, f'leg {leg} missing from legs.tsv'
        bad += [f'leg {leg}: {l.strip()[:90]}' for l in slice_of(ltsm, legs, leg, 'ltsm') if SWAP_PAT.search(l)]
    if bad: return F, '; '.join(bad[:3])
    tr = trace_rows(run)
    if tr is None: return NM, 'no Swap/late recall in the logs, but recognition_trace.csv is absent: MAP new per step not measured'
    rows = [r for leg in FIRST_LEGS for r in tr if legs[leg]['t0_ms'] <= r[0] <= legs[leg]['t1_ms']]
    lost = [r for r in rows if r[1] != 'None' or r[3] < max(r[4].values(), default=0.0)]
    if lost: return F, f'{len(lost)} of {len(rows)} logged steps with MAP not new or an act (first t={lost[0][0]}, p_new={lost[0][3]:.2f})'
    if not rows: return NM, 'trace has no step inside the first-entry legs'
    return P, f'MAP new at {len(rows)}/{len(rows)} logged steps, no Swap'

def c2(legs, ltsm, rooms):
    if RECOG_LEG not in legs: return F, 'leg 4 missing from legs.tsv'
    sl = slice_of(ltsm, legs, RECOG_LEG, 'ltsm')
    sw = [l for l in sl if '[recog] SWAP requested' in l]
    if len(sw) != 1: return F, f'{len(sw)} Swap request(s) in the leg-4 window, expected exactly 1'
    m = re.search(r'room_id (\d+) \(posterior ([0-9.eE+-]+)', sw[0])
    if not m: return F, 'Swap line unparsable: ' + sw[0].strip()[:90]
    rid = int(m.group(1))
    if rooms is not None and rid not in {r[1] for r in rooms}: return F, f'Swap room_id {rid} is not a room of memory'
    c = re.search(r'metric log_inside ([-0-9.eE+]+), appearance loglr ([-0-9.eE+]+)', sw[0])
    if not c: return NM, f'1 Swap to room_id {rid}, posterior {m.group(2)}; the metric/appearance contributions are not on the line'
    return P, f'1 Swap to room_id {rid}, posterior {m.group(2)}, metric log_inside {c.group(1)}, appearance loglr {c.group(2)}'

def c3(rooms, ltsm):
    if rooms is None: return F, 'ltsm_memory.json missing or unreadable'
    names = [strip_id(r[0]) for r in rooms]
    dups = sorted({n for n in names if names.count(n) > 1})
    prov = {m.group(1) for l in ltsm for m in [re.search(r'\[swap\] STAGED: (\S+) is ', l)] if m}
    ghost = sorted(set(names) & prov)
    ok = len(names) == 3 and not dups and not ghost
    msg = f'{len(names)} room node(s): {", ".join(sorted(names))}'
    if dups: msg += f'; duplicate name(s) {", ".join(dups)}'
    if ghost: msg += f'; provisional name(s) still present {", ".join(ghost)}'
    return (P if ok else F), msg

def c4(legs, run):
    p = os.path.join(run, 'gt_error.csv')
    if RECOG_LEG not in legs or not os.path.exists(p): return F, 'leg 4 window or gt_error.csv missing'
    t0, t1 = legs[RECOG_LEG]['t0_ms'], legs[RECOG_LEG]['t1_ms']
    rows = list(csv.DictReader(open(p)))
    if not rows or 'cov_xx' not in rows[0]:
        return NM, 'gt_error.csv has no xy covariance column (only cov_tt): NEES not computable'
    s = []
    for r in rows:
        t = float(r['ts_ms'])
        if t0 <= t <= t1:
            e2 = (float(r['est_x']) - float(r['gt_x'])) ** 2 + (float(r['est_y']) - float(r['gt_y'])) ** 2
            s.append((t, e2 / max(float(r['cov_xx']), 1e-12)))
    if not s: return F, 'no gt_error.csv samples inside the leg-4 window'
    settle = None
    for t, n in reversed(s):          # settled = first sample from which NEES stays <= 5.99 to the end of the leg
        if n > CHI2_2_95: break
        settle = t
    if settle is None: return F, f'NEES above {CHI2_2_95} at the end of leg 4 (last {s[-1][1]:.1f})'
    return P, f'settled {(settle - t0) / 1e3:.1f} s after leg 4 start; NEES <= {CHI2_2_95} for the remaining {len([1 for t, _ in s if t >= settle])} samples'

def c5(sym):
    parent = {}
    for k, n in sym.items():
        if n.get('type') == 'room':
            for l in n.get('links', []):
                if l.get('label') == 'RT': parent[l['dst']] = strip_id(n['name'])
    pairs = []
    for n in sym.values():
        if n.get('type') != 'metaconcept' or (n.get('attribute', {}).get('object_subtype') or {}).get('value') != 'passage': continue
        pairs.append((n['name'], {parent.get(l['dst']) for l in n.get('links', []) if l.get('label') == 'match'}))
    hit = [p for p in pairs if p[1] == {'room_2', 'room_3'}]
    if not hit: return [(F, 'no passage with faces in room_2 and room_3 (passages: ' + ', '.join(f"{a}{sorted(map(str, b))}" for a, b in pairs) + ')')]
    return [(P, f'{hit[0][0]} has faces in room_2 and room_3'),
            (NM, 'seam within 5 cm of the wbt door: memory carries no world frame')]

def c6(legs, ltsm):
    bad = []
    for leg in RETURN_LEGS:
        if leg not in legs or legs[leg]['status'] != 'OK': bad.append(f"leg {leg} not completed ({legs[leg]['status'] if leg in legs else 'absent'})"); continue   # Swaps count only in these windows (leg 4's closes on the Swap line)
        sl = slice_of(ltsm, legs, leg, 'ltsm')
        if not any(re.search(r'\[recall\] \S+ posted', l) for l in sl): bad.append(f'leg {leg}: no recall posted')
        if any(SWAP_PAT.search(l) for l in sl): bad.append(f'leg {leg}: a Swap')
    return (F if bad else P), '; '.join(bad) if bad else 'legs 5 and 6 completed by ordinary passage recall, no Swap'

def c7(ltsm):
    n = sum(1 for l in ltsm if '[recog] UNDO requested' in l or '[swap] UNDONE' in l)
    return (P if n == 0 else F), f'{n} Undo event(s)'

def c8(door):
    late = [(i, m.group(1)) for i, l in enumerate(door) for m in [re.search(r"\[restore\] '([^']+)'.* recalled \(late\)", l)] if m]
    drop = [l for l in door if re.search(r"\[restore\] provisional '[^']+' dropped", l)]
    if not late: return F, 'no "[restore] \'<room>\' recalled (late)" in door_concept.log'
    if not drop: return F, 'recalled (late) but no provisional-room identity drop'
    i, nm = late[0]
    fresh = [l for l in door[i:] if re.search(r"\[restore\] '%s' entered FRESH" % re.escape(nm), l)]
    if fresh: return F, f"'{nm}' entered FRESH after the swap"
    return P, f"'{nm}' recalled (late), provisional identity rows dropped, no FRESH entry after"

def leg5_door(log_path, from_line, door_path=None, door_from=0):
    """The live door to cross back through the recognised room: ltsm's last '[swap] crossed door A <-> K face rN_door_M' line after
    from_line (leg 4 start) -> base name 'door_M', mapped through door_concept's last "[restore] RESTORED as 'ACTUAL' (remembered
    'door_M')" pair after door_from (a live node may have held the name: RestoreRename) -> ACTUAL. '' when ltsm's line is absent
    or says 'new'."""
    last = ''
    for l in read_lines(log_path)[from_line:]:
        m = re.search(r'\[swap\] crossed door (\S+) <-> K face (\S+)', l)
        if m: last = m.group(2)
    m = re.match(r'^r\d+_(door_\d+)', last)
    if not m: return ''
    base = m.group(1)
    if door_path:
        for l in read_lines(door_path)[door_from:]:
            p = re.search(r"\[restore\] RESTORED as '([^']+)' \(remembered '([^']+)'\)", l)
            if p and p.group(2) == base: base = p.group(1)       # last pair wins
    return base

def main():
    if len(sys.argv) in (4, 6) and sys.argv[1] == '--leg5-door':
        a = sys.argv
        print(leg5_door(a[2], int(a[3]), *( (a[4], int(a[5])) if len(a) == 6 else () ))); return 0
    if len(sys.argv) != 2: print(__doc__); return 2
    run = sys.argv[1]
    legs = read_legs(run)
    ltsm = read_lines(os.path.join(run, 'logs', 'ltsm_agent.log'))
    door = read_lines(os.path.join(run, 'logs', 'door_concept.log'))
    rooms, sym = memory_rooms(run)
    res = [('C1', 'first entries (legs 1,3): MAP new, no Swap, no late recall', c1(legs, ltsm, run)),
           ('C2', 'leg 4: exactly one Swap', c2(legs, ltsm, rooms)),
           ('C3', 'no duplicate room', c3(rooms, ltsm)),
           ('C4', 'pose after relocalisation within 2 sigma', c4(legs, run))]
    for i, r in enumerate(c5(sym) if sym else [(F, 'no memory')]):
        res.append(('C5' + 'ab'[i], ('DOOR_2 passage in memory', 'DOOR_2 seam vs wbt')[i], r))
    res += [('C6', 'tour continues by ordinary recall', c6(legs, ltsm)),
            ('C7', 'no wrong commit', c7(ltsm)),
            ('C8', 'door_concept recalls (late), drops provisional identity', c8(door))]
    for k, title, (st, msg) in res: print(f'{st} {k} {title}: {msg}')
    cnt = {x: sum(1 for r in res if r[2][0] == x) for x in (P, F, NM)}
    print(f'{cnt[P]} PASS, {cnt[F]} FAIL, {cnt[NM]} NOT MEASURED')
    return 1 if cnt[F] else 0

if __name__ == '__main__': sys.exit(main())
