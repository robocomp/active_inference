#!/usr/bin/env python3
"""AuRo C3(a): object births and removals on one furnished warm tour, graded against the Webots ground truth.
   c3a_objects.py <run_dir> [--wbt world.wbt]

The rows match the real-robot object table (AuRo paper, figures/table_objects_real.py). That script was not available
here, so every definition below is ours and is printed with the result:

  INSTANCES   ltom_agent's end-of-run repertory (<run>/ltom_repertory_end.json): one entry per object identity (uid),
              with its class, its lives (born_ms, end = death | owner_exit | ...) and keyframes (room frame + room_key).
              The run must start with an EMPTY object memory, so every entry was made in this run.
  POSITION    an instance's last keyframe, taken from its room's frame to the world by a rigid fit per graph room of
              room_concept's pose estimate (gt_error.csv est_x, est_y) to the Webots pose (gt_x_world, gt_y_world).
              The samples come from the time windows in which the robot was in that room (legs.tsv), trimmed by 3 s.
  SEEN        a ground-truth object is in view when it is within RANGE_M of the robot, within +-HFOV_DEG/2 of its heading
              (Webots yaw 0 faces +y) and no wall lies on the line between them (walls of the .wbt, doors open).
              "Seen enough to count" = in view for SEEN_S seconds or more in total.
  MATCH       instance -> the nearest ground-truth object of its OWN class within R[class] m; else the nearest object of
              ANOTHER class within that radius (wrong class); else nothing behind it (phantom).
  HELD        a seen object with at least one matched instance whose last life did not end in a death WHILE PRESENT
              (alive in the graph at the end, or remembered by ltom after the robot left its room).
  REMOVED WHILE PRESENT  a life of a matched instance that ended in `death` while the robot was in the object's room and not
              leaving it (nothing is moved in C3(a)). Deaths during a leg that leaves the room are counted separately:
              chair_concept deletes its nodes on leaving and ltom records that as a death.
  DELAY TO BIRTH  first birth of any instance matched to the object - the first moment the object was in view.
  PLANAR ERROR    final world position of each matched instance vs the object's centre.
"""
import csv, json, math, os, statistics, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import furniture as F
import wbt_rooms as W

RANGE_M, HFOV_DEG, SEEN_S = 4.0, 110.0, 3.0
FIXTURE = 'three_rooms_door0'
R_SCALE = float(os.environ.get('R_SCALE', '1'))   # sensitivity of the verdicts to the match radius
R = {'chair': 0.6, 'bottle': 0.4, 'nightstand': 0.5, 'toilet': 0.6, 'hood': 0.6, 'refrigerator': 0.8,
     'table': 1.0, 'bed': 1.2, 'cabinet': 1.0}


def gt_objects(wbt):
    """[(name, class, x, y)] for every object of a class some concept agent models."""
    t = F.tokens(open(wbt).read()); i = 0; top = []
    while i < len(t):
        if t[i] in ('EXTERNPROTO',): i += 2; continue
        if t[i] == 'DEF': i += 2; continue
        if i + 1 < len(t) and t[i + 1] == '{': n, i = F.parse_node(t, i); top.append(n)
        else: i += 1
    out = []
    def name(n):
        v = n.get('name')
        while isinstance(v, list) and v: v = v[0]
        return v.strip('"') if isinstance(v, str) else ''
    def cls(n):
        ty, nm = n['_type'], name(n)
        if ty == 'Chair': return 'chair'
        if ty == 'Desk' or nm in ('Table1', 'Table1(1)', 'WoodenTable(1)', 'RoundTable'): return 'table'
        if nm == 'Fridge': return 'refrigerator'
        if nm == 'KitchenHood': return 'hood'
        if nm.startswith('Worktop'): return 'cabinet'
        if ty == 'BeerBottle': return 'bottle'
        if ty == 'Bed': return 'bed'
        if nm.startswith('nightstand'): return 'nightstand'
        if ty == 'Toilet': return 'toilet'
        return None
    def visit(n):
        if n['_type'] == 'Group':
            for c in n.get('children', []):
                if isinstance(c, dict): visit(c)
            return
        c = cls(n)
        if c:
            x, y, _, _ = F.T(n)
            out.append((name(n) or n['_type'], c, x, y))
    for n in top: visit(n)
    return out


def rigid_fit(A, B):
    """B ~ Rot A + t (2-D, least squares). Returns (theta, t, rms)."""
    A, B = np.asarray(A), np.asarray(B)
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    th = math.atan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
    Rm = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    t = cb - Rm @ ca
    rms = float(np.sqrt(np.mean(np.sum((A @ Rm.T + t - B) ** 2, 1))))
    return th, t, rms


def main(run, wbt):
    legs = {int(r['leg']): (int(r['t0_ms']), int(r['t1_ms']), r['arg'], r['status'])
            for r in csv.DictReader(open(f'{run}/legs.tsv'), delimiter='\t')}
    gt = [(int(r['t_ms']), float(r['x']), float(r['y']), float(r['yaw']))
          for r in csv.DictReader(open(f'{run}/gt_pose.tsv'), delimiter='\t')]
    t_start, t_end = gt[0][0], gt[-1][0]
    # room windows from the legs: before leg 1 = room_1 (the fixture's start room), after leg k = its arg
    wins, prev_t, prev_room = [], t_start, 'room_1'
    for k in sorted(legs):
        t0, t1, arg, st = legs[k]
        wins.append((prev_room, prev_t, t0))
        prev_t, prev_room = t1, arg
    wins.append((prev_room, prev_t, t_end))
    est = [(int(r['ts_ms']), float(r['est_x']), float(r['est_y']), float(r['gt_x_world']), float(r['gt_y_world']))
           for r in csv.DictReader(open(f'{run}/gt_error.csv'))]
    # ONE FIT PER VISIT, not per room: a room's frame can move between visits (c3b_on_ltom1: room_1 rotated 0.07 rad and
    # shifted 0.5 m between the start and the end of the tour), and a keyframe is in the frame of the moment it was taken.
    fits = []   # (room, a, b, theta, t, rms)
    for rm, a, b in wins:
        A = [(ex, ey) for t, ex, ey, gx, gy in est if a + 3000 <= t <= b - 3000]
        B = [(gx, gy) for t, ex, ey, gx, gy in est if a + 3000 <= t <= b - 3000]
        if len(A) >= 10: fits.append((rm, a, b) + rigid_fit(A, B))
    # est_x/est_y are in the MAP frame (room_1's), whichever room the robot is in; an object keyframe is in its OWN room's
    # frame. Each room's pose in the map frame comes from the LTSM layout (the run's newest *_layout.json, else the
    # fixture's): room -> map (layout) -> world (the fit of the visit nearest the keyframe's time).
    import glob
    lay = sorted(glob.glob(f'{run}/ltsm_history/*_layout.json')) or sorted(glob.glob(f'{os.path.dirname(run)}/fixtures/{FIXTURE}/ltsm_history/*_layout.json'))
    room_pose = {}
    for r in json.load(open(lay[-1]))['rooms']:
        room_pose['room_' + r['name'].rsplit('_', 1)[-1]] = (r['x'], r['y'], r['yaw'])
    print(f'  room poses in the map frame from {lay[-1]}: ' + ', '.join(f'{k} ({v[0]:.2f},{v[1]:.2f},{v[2]:.3f})' for k, v in sorted(room_pose.items())))
    def to_world(room, x, y, t_ms):
        if room not in room_pose or not fits: return None
        rx, ry, ryaw = room_pose[room]
        mx, my = rx + math.cos(ryaw) * x - math.sin(ryaw) * y, ry + math.sin(ryaw) * x + math.cos(ryaw) * y
        f = min(fits, key=lambda f: 0 if f[1] <= t_ms <= f[2] else min(abs(t_ms - f[1]), abs(t_ms - f[2])))
        th, t = f[3], f[4]
        return (math.cos(th) * mx - math.sin(th) * my + t[0], math.sin(th) * mx + math.cos(th) * my + t[1])

    objs = gt_objects(wbt)
    # only classes whose agent RAN (agents.txt from concept_agents.sh): an object nobody models is not "never born"
    ran = {l.split('\t')[0].removesuffix('_concept') for l in open(f'{run}/agents.txt') if l.strip()} if os.path.exists(f'{run}/agents.txt') else None
    if ran:
        dropped = sorted({o[1] for o in objs} - ran)
        objs = [o for o in objs if o[1] in ran]
        if dropped: print(f'  classes with no agent in this run, not graded: {", ".join(dropped)}')
    walls, doors, meshes = W.parse_wbt(wbt)
    occ = W.rasterise(walls, {k: dict(v, width=0.0) for k, v in doors.items()}, meshes)
    x0, _, y0, _ = W.BOX
    def blocked(a, b):
        n = int(math.hypot(b[0] - a[0], b[1] - a[1]) / 0.02)
        for i in range(2, n - 2):
            x = a[0] + (b[0] - a[0]) * i / n; y = a[1] + (b[1] - a[1]) * i / n
            if occ[int((y - y0) / W.RES), int((x - x0) / W.RES)]: return True
        return False
    seen_s = [0.0] * len(objs); first_seen = [None] * len(objs)
    def samples():   # the GT pose is logged every ~1.5 s; a turning robot sweeps its field of view between rows
        for a, b in zip(gt, gt[1:]):
            n = max(1, int((b[0] - a[0]) / 100))
            dyaw = math.remainder(b[3] - a[3], 2 * math.pi)
            for k in range(n):
                f = k / n
                yield (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2]), a[3] + f * dyaw,
                       (b[0] - a[0]) / n / 1000.0)
    for t, x, y, yaw, dt in samples():
        if dt > 2.0: continue
        fwd = yaw + math.pi / 2
        for j, (_, _, ox, oy) in enumerate(objs):
            d = math.hypot(ox - x, oy - y)
            if d > RANGE_M: continue
            if abs(math.remainder(math.atan2(oy - y, ox - x) - fwd, 2 * math.pi)) > math.radians(HFOV_DEG / 2): continue
            if blocked((x, y), (ox, oy)): continue
            seen_s[j] += dt
            if first_seen[j] is None: first_seen[j] = t

    rep = json.load(open(f'{run}/ltom_repertory_end.json'))
    inst = []
    stale = 0
    for e in rep['entries']:
        # guard: only what THIS run made. A memory that did not start empty (11-21: ltom loads .prev when the file is
        # missing) leaves older lives; they are dropped and counted, never graded.
        kf = [k for k in e.get('keyframes', []) if k['t_ms'] >= t_start] or e.get('keyframes', [])  # recalled, never re-keyed: its remembered place
        e = dict(e, lives=[l for l in e.get('lives', []) if l['born_ms'] >= t_start], keyframes=kf)
        if not e['lives']: stale += 1; continue
        if not e['keyframes']: print(f"  ⚠ {e['name']}: a life in this run but no keyframe at all — not graded"); continue
        k = e['keyframes'][-1]; room = f"room_{k['room_key']}"
        p = to_world(room, k['x'], k['y'], k['t_ms'])
        if p is None: continue
        wx, wy = p
        c = e['cls']; best = None
        for j, (nm, oc, ox, oy) in enumerate(objs):
            d = math.hypot(ox - wx, oy - wy)
            if d <= R_SCALE * R.get(c, 0.75) and (best is None or (oc == c, -d) > (objs[best[0]][1] == c, -best[1])): best = (j, d)
        kind = 'phantom' if best is None else ('match' if objs[best[0]][1] == c else 'wrong_class')
        inst.append(dict(uid=e['uid'], name=e['name'], cls=c, room=room, wx=wx, wy=wy, kind=kind,
                         obj=None if best is None else best[0], err=None if best is None else best[1],
                         lives=e['lives'], born=min(l['born_ms'] for l in e['lives']),
                         last_end=e['lives'][-1].get('end', '')))

    # CABINETS NEVER REACH LTOM (c3a_cab1: 0 cabinet entries although `cabinet` is in ltom's classes), so they are read from
    # cabinet_concept's own log: "[kitchen] BIRTH '<name>' id=<node> run=(cx,cy)" = the run's centre in its room's frame
    # (the RT translation it writes). One instance per cabinet NAME (a run keeps its name across rebirths); one life per
    # node insert (cabinet births are LiDAR-driven, so a negative delay to birth is possible: the camera-cone "in view" rule
    # does not bound them), ended by the trace's node_delete of that node (none = alive at the end). Room = the RT edge's parent.
    import re
    cab_log = f'{run}/logs/cabinet_concept.log'
    if os.path.exists(cab_log) and (ran is None or 'cabinet' in ran):
        parent, deleted = {}, {}
        for r in csv.DictReader(open(f'{run}/graph_trace.csv')):
            if r['event'] == 'edge_insert' and r['edge_type'] == 'RT' and r['a_type'] == 'room': parent[r['b_id']] = r['a_name']
            if r['event'] == 'node_delete': deleted[r['a_id']] = int(r['t_ms'])
        cabs = {}
        for l in open(cab_log, errors='replace'):
            m = re.match(r"^(\d+\.\d+) cabinet_concept: \[kitchen\] BIRTH '([^']+)' id=(\d+) run=\(([-\d.]+),([-\d.]+)\)", l)
            if not m: continue
            t, name, nid, cx, cy = int(float(m.group(1)) * 1000), m.group(2), m.group(3), float(m.group(4)), float(m.group(5))
            room = parent.get(nid)
            if room is None: continue
            end = deleted.get(nid)
            life = {'born_ms': t, 'end': 'death' if end is not None and end < t_end - 5000 else 'owner_exit' if end is not None else 'alive',
                    'end_ms': end if end is not None else -1}
            c = cabs.setdefault((name, room), {'name': name, 'lives': []})   # names repeat across rooms (w0_base in room_2 AND room_3)
            c['lives'].append(life); c.update(room=room, x=cx, y=cy, t=t)
        for c in cabs.values():
            p = to_world(c['room'], c['x'], c['y'], c['t'])
            if p is None: continue
            wx, wy = p; best = None
            for j, (nm, oc, ox, oy) in enumerate(objs):
                d = math.hypot(ox - wx, oy - wy)
                if d <= R_SCALE * R['cabinet'] and (best is None or (oc == 'cabinet', -d) > (objs[best[0]][1] == 'cabinet', -best[1])): best = (j, d)
            kind = 'phantom' if best is None else ('match' if objs[best[0]][1] == 'cabinet' else 'wrong_class')
            inst.append(dict(uid=f"log:{c['name']}@{c['room']}", name=c['name'], cls='cabinet', room=c['room'], wx=wx, wy=wy, kind=kind,
                             obj=None if best is None else best[0], err=None if best is None else best[1], lives=c['lives'],
                             born=min(l['born_ms'] for l in c['lives']), last_end=c['lives'][-1]['end']))
    # A `death` is ltom's end of a life with the node deleted. chair_concept (at least) deletes its nodes when the robot LEAVES
    # the room, and ltom records that as a death too (c3b_on_ltom1: every chair death fell on a leg start). So a death is
    #   at_exit  during a leg that leaves the object's room (t0 - 3 s .. t1 + 3 s)   -> not a removal
    #   present  while the robot is in the object's room                             -> REMOVED WHILE PRESENT
    #   away     while the robot is in another room                                  -> reported, not counted as present
    def death_kind(room, t):
        for k in sorted(legs):
            t0, t1, arg, _ = legs[k]
            frm = 'room_1' if k == min(legs) else legs[k - 1][2]
            if frm == room and t0 - 3000 <= t <= t1 + 3000: return 'at_exit'
        for rm, a, b in wins:
            if a <= t <= b: return 'present' if rm == room else 'away'
        return 'away'
    for i in inst:
        i['deaths'] = [death_kind(i['room'], l['end_ms']) for l in i['lives'] if l.get('end') == 'death']
        i['gone'] = bool(i['lives']) and i['lives'][-1].get('end') == 'death' and i['deaths'][-1] == 'present'
    with open(f'{run}/c3a_instances.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['uid', 'name', 'class', 'room', 'x_world', 'y_world', 'verdict', 'gt_object', 'gt_class', 'planar_error_m',
                    'n_lives', 'deaths_present', 'deaths_at_exit', 'deaths_away', 'last_end', 'first_born_ms', 'first_born_from_start_s'])
        for i in inst:
            o = objs[i['obj']] if i['obj'] is not None else None
            w.writerow([i['uid'], i['name'], i['cls'], i['room'], f"{i['wx']:.3f}", f"{i['wy']:.3f}", i['kind'],
                        o[0] if o else '', o[1] if o else '', f"{i['err']:.3f}" if i['err'] is not None else '',
                        len(i['lives']), i['deaths'].count('present'), i['deaths'].count('at_exit'), i['deaths'].count('away'), i['last_end'], i['born'],
                        f"{(i['born'] - t_start) / 1000:.1f}"])
    rows = []
    for j, (nm, c, ox, oy) in enumerate(objs):
        mine = [i for i in inst if i['obj'] == j and i['kind'] == 'match']
        held = any(not i['gone'] for i in mine)
        born = min((i['born'] for i in mine), default=None)
        delay = (born - first_seen[j]) / 1000 if born is not None and first_seen[j] is not None else None
        rows.append(dict(name=nm, cls=c, x=ox, y=oy, seen_s=seen_s[j], seen=seen_s[j] >= SEEN_S, n_inst=len(mine),
                         lives=sum(len(i['lives']) for i in mine), deaths=sum(i['deaths'].count('present') for i in mine),
                         deaths_exit=sum(i['deaths'].count('at_exit') for i in mine), deaths_away=sum(i['deaths'].count('away') for i in mine),
                         held=held, delay=delay, errs=[i['err'] for i in mine]))
    with open(f'{run}/c3a_objects.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['gt_object', 'class', 'x', 'y', 'seen_s', 'seen_enough', 'instances', 'lives', 'deaths', 'held_at_end',
                    'deaths_at_exit', 'deaths_away', 'delay_to_birth_s', 'planar_error_m'])
        for r in rows:
            w.writerow([r['name'], r['cls'], f"{r['x']:.3f}", f"{r['y']:.3f}", f"{r['seen_s']:.1f}", int(r['seen']), r['n_inst'],
                        r['lives'], r['deaths'], int(r['held']), r['deaths_exit'], r['deaths_away'], '' if r['delay'] is None else f"{r['delay']:.1f}",
                        '|'.join(f'{e:.3f}' for e in r['errs'])])

    seen = [r for r in rows if r['seen']]; held = [r for r in seen if r['held']]
    med = lambda v: f'{statistics.median(v):.2f}' if v else 'n/a'
    print(f'run {run}')
    print(f'  visit fits (est->world): ' + ', '.join(f'{f[0]} rms {f[5]:.3f} m' for f in fits))
    if stale: print(f'  ⚠ {stale} ltom entries had no life in this run (memory did not start empty) — ignored')
    print(f'  definitions: range {RANGE_M} m, HFOV {HFOV_DEG} deg, seen >= {SEEN_S} s, match radius by class {R}')
    print(f'  objects seen enough to count     {len(seen)} (of {len(rows)} modelled ground-truth objects)')
    print(f'  objects held at the end          {len(held)} ({len(held) / max(1, len(seen)):.2f})')
    print(f'  instances per held object        {statistics.mean(r["n_inst"] for r in held):.2f}' if held else '  instances per held object        n/a')
    print(f'  objects removed while present    {sum(r["deaths"] > 0 for r in rows)} objects, {sum(r["deaths"] for r in rows)} deaths '
          f'(not counted: {sum(r["deaths_exit"] for r in rows)} at a room exit, {sum(r["deaths_away"] for r in rows)} while away)')
    print(f'  instances on another class       {sum(i["kind"] == "wrong_class" for i in inst)}')
    print(f'  phantoms                         {sum(i["kind"] == "phantom" for i in inst)}')
    print(f'  median delay to birth (s)        {med([r["delay"] for r in seen if r["delay"] is not None])}')
    print(f'  median planar error (m)          {med([e for r in rows for e in r["errs"]])}')
    print('  per class: class | objects (seen/all) | held | lives per held object | phantoms | wrong class | never born | median time to birth s')
    for c in sorted({r['cls'] for r in rows}):
        rc = [r for r in rows if r['cls'] == c]; sc = [r for r in rc if r['seen']]; hc = [r for r in sc if r['held']]
        ic = [i for i in inst if i['cls'] == c]
        print(f"    {c:12s} | {len(sc)}/{len(rc)} | {len(hc)} | {statistics.mean(r['lives'] for r in hc):.2f} | " if hc else f"    {c:12s} | {len(sc)}/{len(rc)} | 0 | n/a | ", end='')
        print(f"{sum(i['kind'] == 'phantom' for i in ic)} | {sum(i['kind'] == 'wrong_class' for i in ic)} | "
              f"{sum(r['n_inst'] == 0 for r in sc)} | {med([r['delay'] for r in sc if r['delay'] is not None])}")
    print(f'  csv: {run}/c3a_objects.csv, {run}/c3a_instances.csv')


if __name__ == '__main__':
    wbt = W.WBT
    if '--wbt' in sys.argv: wbt = sys.argv[sys.argv.index('--wbt') + 1]
    main(sys.argv[1].rstrip('/'), wbt)
