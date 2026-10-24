#!/usr/bin/env python3
"""Room-recognition gold tour report (spec 2026-10-22).  python3 recognition_report.py <run_dir>

Prints PASS/FAIL per criterion C1..C8 and exits 1 on any FAIL.  Tour legs (recognition_tour.sh): 1 first entry of a new
room, 2 back, 3 first entry of a second new room, 4 the RECOGNITION leg (new door into a remembered room), 5 back through
the carried passage, 6 on to room_1 (ordinary recalls).

Inputs in <run_dir>:
  legs.tsv                 driver: leg, status, t0_ms, t1_ms (epoch), and the 0-based line range [x0,x1) each log held
                           during the leg (ltsm0/1, door0/1, room0/1); the agent logs carry no wall clock.
  logs/ltsm_agent.log, logs/door_concept.log, logs/room_concept.log
  ltsm_memory.json         copy of the ltsm DSR persistence at the end of the tour
  gt_error.csv             ts_ms, gt_x, gt_y, est_x, est_y and cov_xx (the localiser's xy variance, m^2, isotropic)
  recognition_trace.csv    OPTIONAL (leg, t_ms, p_new): per-step MAP check for C1.  Without it C1 is inferred from the
                           absence of any Swap/late recall in the leg's log slice.
Log text parsed (ltsm room_recognizer.cpp / room_swap.cpp / room_recall.cpp, door_concept specificworker.cpp):
  "[recog] SWAP requested: room_id N (posterior p)", "[swap] STAGED: P is <mem> (case ..", "[recog] UNDO requested",
  "[swap] UNDONE", "[recall] LATE ..", "[recall] recall_room_N posted", "[restore] 'X' recalled (late)",
  "[restore] provisional 'X' dropped", "[restore] 'X' entered FRESH", "[room][late-recall] X adopted".
"""
import csv, json, math, os, re, sys

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

def c1(legs, ltsm, run):
    bad, n = [], 0
    for leg in FIRST_LEGS:
        if leg not in legs: return False, f'leg {leg} missing from legs.tsv'
        sl = slice_of(ltsm, legs, leg, 'ltsm')
        n += 1
        bad += [f'leg {leg}: {l.strip()[:90]}' for l in sl if SWAP_PAT.search(l)]
    tr = os.path.join(run, 'recognition_trace.csv'); how = 'inferred from no Swap/late recall (no per-step posterior log)'
    if os.path.exists(tr):
        rows = [r for r in csv.DictReader(open(tr)) if int(r['leg']) in FIRST_LEGS]
        lost = [r for r in rows if float(r['p_new']) < 0.5]
        bad += [f"leg {r['leg']} t={r['t_ms']}: p_new={float(r['p_new']):.2f} < MAP" for r in lost]
        how = f'MAP new at {len(rows) - len(lost)}/{len(rows)} logged steps, no Swap'
    return not bad, (how if not bad else '; '.join(bad[:3]))

def c2(legs, ltsm, rooms):
    if RECOG_LEG not in legs: return False, 'leg 4 missing from legs.tsv'
    sl = slice_of(ltsm, legs, RECOG_LEG, 'ltsm')
    sw = [l for l in sl if '[recog] SWAP requested' in l]
    if len(sw) != 1: return False, f'{len(sw)} Swap request(s) in leg 4, expected exactly 1'
    m = re.search(r'room_id (\d+) \(posterior ([0-9.eE+-]+)\)', sw[0])
    if not m: return False, 'Swap line unparsable: ' + sw[0].strip()[:90]
    rid = int(m.group(1))
    st = [l for l in sl if '[swap] STAGED' in l]
    if rooms is not None and rid not in {r[1] for r in rooms}: return False, f'Swap room_id {rid} is not a room of memory'
    contrib = re.search(r'metric[^,;)]*|appearance[^,;)]*', sw[0] + ' '.join(st))
    return True, (f'1 Swap to room_id {rid}, posterior {m.group(2)}; ' +
                  (contrib.group(0) if contrib else 'metric/appearance contributions not in the log text'))

def c3(rooms, ltsm):
    if rooms is None: return False, 'ltsm_memory.json missing or unreadable'
    names = [strip_id(r[0]) for r in rooms]
    dups = sorted({n for n in names if names.count(n) > 1})
    prov = {m.group(1) for l in ltsm for m in [re.search(r'\[swap\] STAGED: (\S+) is ', l)] if m}
    ghost = sorted(set(names) & prov)
    ok = len(names) == 3 and not dups and not ghost
    msg = f'{len(names)} room node(s): {", ".join(sorted(names))}'
    if dups: msg += f'; duplicate name(s) {", ".join(dups)}'
    if ghost: msg += f'; provisional name(s) still present {", ".join(ghost)}'
    return ok, msg + ('' if ok or len(names) != 3 else '')

def c4(legs, run):
    p = os.path.join(run, 'gt_error.csv')
    if RECOG_LEG not in legs or not os.path.exists(p): return False, 'leg 4 window or gt_error.csv missing'
    t0, t1 = legs[RECOG_LEG]['t0_ms'], legs[RECOG_LEG]['t1_ms']
    rows = list(csv.DictReader(open(p)))
    if not rows or 'cov_xx' not in rows[0]:
        return False, 'gt_error.csv has no cov_xx column (localiser xy covariance): NEES not computable'
    s = []
    for r in rows:
        t = float(r['ts_ms'])
        if t0 <= t <= t1:
            e2 = (float(r['est_x']) - float(r['gt_x'])) ** 2 + (float(r['est_y']) - float(r['gt_y'])) ** 2
            s.append((t, e2 / max(float(r['cov_xx']), 1e-12)))
    if not s: return False, 'no gt_error.csv samples inside the leg-4 window'
    settle = None
    for t, n in reversed(s):          # settled = first sample from which NEES stays <= 5.99 to the end of the leg
        if n > CHI2_2_95: break
        settle = t
    if settle is None: return False, f'NEES above {CHI2_2_95} at the end of leg 4 (last {s[-1][1]:.1f})'
    return True, f'settled {(settle - t0) / 1e3:.1f} s after leg 4 start; NEES <= {CHI2_2_95} for the remaining {len([1 for t, _ in s if t >= settle])} samples'

def c5(sym):
    names = {k: n['name'] for k, n in sym.items()}
    parent = {}
    for k, n in sym.items():
        if n.get('type') == 'room':
            for l in n.get('links', []):
                if l.get('label') == 'RT': parent[l['dst']] = strip_id(n['name'])
    pairs = []
    for n in sym.values():
        if n.get('type') != 'metaconcept' or (n.get('attribute', {}).get('object_subtype') or {}).get('value') != 'passage': continue
        rs = {parent.get(l['dst']) for l in n.get('links', []) if l.get('label') == 'match'}
        pairs.append((n['name'], rs))
    hit = [p for p in pairs if p[1] == {'room_2', 'room_3'}]
    if not hit: return False, 'no passage with faces in room_2 and room_3 (passages: ' + ', '.join(f"{a}{sorted(map(str, b))}" for a, b in pairs) + ')'
    return True, f'{hit[0][0]} has faces in room_2 and room_3 (seam vs wbt door within 5 cm: not checked here, memory carries no world frame)'

def c6(legs, ltsm):
    bad = []
    for leg in RETURN_LEGS:
        if leg not in legs or legs[leg]['status'] != 'OK': bad.append(f'leg {leg} not completed'); continue
        sl = slice_of(ltsm, legs, leg, 'ltsm')
        if not any(re.search(r'\[recall\] \S+ posted', l) for l in sl): bad.append(f'leg {leg}: no recall posted')
        if any(SWAP_PAT.search(l) for l in sl): bad.append(f'leg {leg}: a Swap')
    return not bad, '; '.join(bad) if bad else 'legs 5 and 6 completed by ordinary passage recall, no Swap'

def c7(ltsm):
    n = sum(1 for l in ltsm if '[recog] UNDO requested' in l or '[swap] UNDONE' in l)
    return n == 0, f'{n} Undo event(s)'

def c8(door):
    late = [(i, m.group(1)) for i, l in enumerate(door) for m in [re.search(r"\[restore\] '([^']+)'.* recalled \(late\)", l)] if m]
    drop = [l for l in door if re.search(r"\[restore\] provisional '[^']+' dropped", l)]
    if not late: return False, 'no "[restore] \'<room>\' recalled (late)" in door_concept.log'
    if not drop: return False, 'recalled (late) but no provisional-room identity drop'
    i, nm = late[0]
    fresh = [l for l in door[i:] if re.search(r"\[restore\] '%s' entered FRESH" % re.escape(nm), l)]
    if fresh: return False, f"'{nm}' entered FRESH after the swap"
    return True, f"'{nm}' recalled (late), provisional identity rows dropped, no FRESH entry after"

def main():
    if len(sys.argv) != 2: print(__doc__); return 2
    run = sys.argv[1]
    legs = read_legs(run)
    ltsm = read_lines(os.path.join(run, 'logs', 'ltsm_agent.log'))
    door = read_lines(os.path.join(run, 'logs', 'door_concept.log'))
    rooms, sym = memory_rooms(run)
    res = [('C1', 'first entries (legs 1,3): MAP new, no Swap, no late recall', c1(legs, ltsm, run)),
           ('C2', 'leg 4: exactly one Swap', c2(legs, ltsm, rooms)),
           ('C3', 'no duplicate room', c3(rooms, ltsm)),
           ('C4', 'pose after relocalisation within 2 sigma', c4(legs, run)),
           ('C5', 'DOOR_2 passage in memory', c5(sym) if sym else (False, 'no memory')),
           ('C6', 'tour continues by ordinary recall', c6(legs, ltsm)),
           ('C7', 'no wrong commit', c7(ltsm)),
           ('C8', 'door_concept recalls (late), drops provisional identity', c8(door))]
    bad = 0
    for k, title, (ok, msg) in res:
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'} {k} {title}: {msg}")
    print(f'{len(res) - bad}/{len(res)} criteria pass')
    return 1 if bad else 0

if __name__ == '__main__': sys.exit(main())
