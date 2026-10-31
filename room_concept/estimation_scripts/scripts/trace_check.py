#!/usr/bin/env python3
"""trace_check.py -- evaluate the room-transition protocol's trace-checkable invariants on a recorded graph trace.

The TLA+ model (docs/superpowers/specs/tla/RoomTransition.tla) states the protocol as guarded actions over graph facts;
this script replays what graph_recorder logged (graph_trace.csv) and checks the same properties on a REAL run.

Input: graph_trace.csv  t_ms,event,agent,a_id,a_name,a_type,b_id,b_name,edge_type,attr,value
       events node_insert,node_rename,node_delete,edge_insert,edge_delete,attr,attr_birth,snap_node,snap_edge
       agent = the writer's DSR agent id (room_concept 5, door_concept 15, ltsm_agent 19, controller 8, residual 14,
       robot_concept 6, retina 9, table_concept 7; 0 = the snapshot taken when the recorder joined).
       The recorder logs NO edge_delete for an edge removed implicitly because an endpoint node was deleted (edges INTO
       the deleted node in particular): every node_delete drops all edges touching that node here.
Optional ground truth: gt_pose.tsv (t_ms x y yaw ..., 1 Hz) + the Webots world (DOOR_k translation/rotation/size), for
       the bounded liveness check "the robot physically past a door ~> a room-change fact within --bound seconds".

Safety (each violation printed with its timestamp):
  S1 exec-door      a door node is deleted while an `executing` edge targets its aff_<door>_cross. Classified:
                    BEFORE-SWITCH (`current` still on the door's room: the door the robot is crossing vanished -- bug 1 of
                    recog_proto11) or AFTER-SWITCH (`current` already moved: the room's owner-less eviction took it).
  S2 owner          a node deleted / renamed by an agent other than the one that inserted it (rule 1); and the protocol
                    edges written by a non-owner (`current` ltsm, `proto` room_concept, `match` ltsm, `executing`
                    controller).
  S3 one-current    more than one robot->room `current` edge live at once.
  S4 names          two live room nodes with one name.
  S5 passage        a live passage_N whose two `match`ed doors hang from the SAME room.
  S6 recall-live    recall_<room> inserted while a live room carries that name (room_recall.cpp:162 says never).
  S7 false-switch  (gt) a room_concept switch that no true door crossing explains (crossings matched one-to-one).
  S8 leg-without-crossing (gt) a controller_mission leg went in_progress -> completed with no true crossing between.
  C1 recall-churn   the same recall_<room> node inserted >= 3 times (ltsm re-posting it: RoomSwap staging, proto15).
  C2 proposal-churn the same recognition proposal rejected >= 3 times.
Liveness (bounded):
  L1 crossing       a true door-line crossing (gt + world) with no room-change fact (room_concept's robot->room RT
                    insert, or ltsm's `current` insert) in [t - 2 s, t + bound].
  L2 current        room_concept's robot->room RT to a new room not followed by `current` on it within bound.
Exit status 1 if any violation was found.

Usage: trace_check.py RUN_DIR [--trace graph_trace.csv] [--gt gt_pose.tsv] [--world threerooms-piso.wbt] [--bound 20]
"""
import argparse, collections, csv, math, os, re, sys

AGENTS = {0: 'snapshot', 5: 'room_concept', 6: 'robot_concept', 7: 'table_concept', 8: 'controller', 9: 'retina',
          14: 'residual', 15: 'door_concept', 19: 'ltsm_agent'}
EDGE_OWNER = {'current': 19, 'proto': 5, 'match': 19, 'executing': 8}
DEFAULT_WORLD = os.path.expanduser('~/robocomp/components/webots-shadow/worlds/piso/threerooms-piso.wbt')


def ag(a):
    return f"{AGENTS.get(a, 'agent')}({a})"


class Graph:
    def __init__(s):
        s.nodes = {}                       # id -> dict(name, type, owner)
        s.edges = {}                       # (from, to, type) -> writer
        s.out = collections.defaultdict(set)
        s.inn = collections.defaultdict(set)

    def add_node(s, i, name, typ, owner):
        s.nodes[i] = {'name': name, 'type': typ, 'owner': owner}

    def del_node(s, i):
        s.nodes.pop(i, None)
        for k in list(s.out.pop(i, set())) + list(s.inn.pop(i, set())):
            s.edges.pop(k, None)
            s.out[k[0]].discard(k); s.inn[k[1]].discard(k)

    def add_edge(s, f, t, typ, w):
        k = (f, t, typ); s.edges[k] = w; s.out[f].add(k); s.inn[t].add(k)

    def del_edge(s, f, t, typ):
        k = (f, t, typ); s.edges.pop(k, None); s.out[f].discard(k); s.inn[t].discard(k)

    def live_edge(s, k):
        return k in s.edges and k[0] in s.nodes and k[1] in s.nodes

    def rt_parent(s, i):
        for k in s.inn.get(i, ()):
            if k[2] == 'RT' and k[0] in s.nodes:
                return k[0]
        return None

    def room_of(s, i):
        for _ in range(16):
            if i is None or i not in s.nodes: return None
            if s.nodes[i]['type'] == 'room': return i
            i = s.rt_parent(i)
        return None

    def name(s, i):
        return s.nodes[i]['name'] if i in s.nodes else f'<{i}>'

    def currents(s):
        return [k for k in s.edges if k[2] == 'current' and s.live_edge(k) and s.nodes[k[1]]['type'] == 'room']


def read_trace(path):
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            r['t'] = int(r['t_ms']); r['agent'] = int(r['agent'] or 0)
            r['a'] = int(r['a_id']) if r['a_id'] else None
            r['b'] = int(r['b_id']) if r['b_id'] else None
            yield r


def check_trace(path, bound_ms):
    G = Graph()
    viol = collections.defaultdict(list)
    room_changes = []                     # (t, kind, room id, room name, agent)
    rt_new = []                           # (t, room id, name): room_concept's robot->room RT inserts
    cur_ins = []                          # (t, room id)
    snap_edges = []
    stopping = {}                         # agent -> t its agent node was deleted (graceful stop)
    info = []
    exec_lost = {}                        # door name -> (t, deleter, executor) when its aff_<door>_cross went executing
    extra = {'mission': [], 'recall_ins': [], 'proposal': []}
    for r in read_trace(path):
        ev, t, w, a, b, et = r['event'], r['t'], r['agent'], r['a'], r['b'], r['edge_type']
        if ev in ('attr', 'attr_birth') and r['attr'] == 'state':
            if r['a_name'] == 'controller_mission': extra['mission'].append((t, r['value']))
            if r['a_name'].startswith('proposal_'): extra['proposal'].append((t, r['a_name'], r['value'], w))
        if ev == 'node_insert' and r['a_name'].startswith('recall_'): extra['recall_ins'].append((t, r['a_name']))
        if ev == 'snap_node':
            G.add_node(a, r['a_name'], r['a_type'], 0)
        elif ev == 'snap_edge':
            snap_edges.append((a, b, et))
        elif ev == 'node_insert':
            if snap_edges:                # the snapshot's edges, once its nodes are known (dangling ones never live)
                for f, to, typ in snap_edges: G.add_edge(f, to, typ, 0)
                snap_edges = []
            if r['a_type'] == 'room' and any(n['type'] == 'room' and n['name'] == r['a_name'] for n in G.nodes.values()):
                viol['S4 names'].append(f"t={t} room '{r['a_name']}' inserted by {ag(w)} while a live room has that name")
            if r['a_name'].startswith('recall_'):
                k = r['a_name'][len('recall_'):]
                if any(n['type'] == 'room' and n['name'] == k for n in G.nodes.values()):
                    viol['S6 recall-live'].append(f"t={t} {r['a_name']} posted by {ag(w)} while room '{k}' is live")
            G.add_node(a, r['a_name'], r['a_type'], w)
        elif ev == 'node_rename':
            n = G.nodes.get(a)
            if n is not None:
                if n['owner'] not in (0, w):
                    viol['S2 owner'].append(f"t={t} {ag(w)} renamed {n['type']} '{n['name']}' -> '{r['b_name'] or r['value']}' owned by {ag(n['owner'])}")
                n['name'] = r['b_name'] or r['value'] or n['name']
        elif ev == 'node_delete':
            n = G.nodes.get(a)
            if n is None: continue
            if n['type'] == 'agent' and n['owner'] == w:
                stopping[w] = t           # the agent's own node goes first on a graceful stop (SIGTERM cleanup)
            if n['owner'] not in (0, w):
                viol['S2 owner'].append(f"t={t} {ag(w)} deleted {n['type']} '{n['name']}' owned by {ag(n['owner'])}")
            m = re.fullmatch(r'aff_(door_\d+)_cross', n['name'])
            if m:   # the executing edge dies with this node; remember it for the door's removal that follows
                for k in [k for k in G.inn.get(a, ()) if k[2] == 'executing' and G.live_edge(k)]:
                    exec_lost[m.group(1)] = (t, w, k[0])
            if n['type'] == 'object' and re.fullmatch(r'door_\d+', n['name']):
                aff = 'aff_' + n['name'] + '_cross'
                execs = [k[0] for k in G.edges if k[2] == 'executing' and G.live_edge(k) and G.name(k[1]) == aff]
                # door_concept's teardown removes the door's affordances FIRST (door_existence.cpp:1679-1699), which
                # silently drops the consumer's `executing` edge a millisecond before the door: same removal.
                if n['name'] in exec_lost and exec_lost[n['name']][1] == w and t - exec_lost[n['name']][0] <= 1000:
                    execs.append(exec_lost[n['name']][2])
                if w in stopping and t - stopping[w] <= 5000:
                    if execs: info.append(f"t={t} {n['name']} removed by {ag(w)} during its shutdown teardown "
                                          f"(`executing` was live) -- not counted")
                    execs = []
                for ex in execs:
                    room = G.room_of(a)
                    cur = [c[1] for c in G.currents()]
                    before = room is not None and room in cur
                    cls = 'BEFORE-SWITCH (the robot is crossing it; bug 1)' if before else \
                          'AFTER-SWITCH (`current` already left its room)'
                    viol['S1 exec-door'].append(
                        f"t={t} {n['name']} (room {G.name(room) if room else '?'}) deleted by {ag(w)} while "
                        f"{G.name(ex)} -> {aff} `executing` was live; current = {[G.name(c) for c in cur]} -- {cls}")
            G.del_node(a)
        elif ev == 'edge_insert':
            if et in EDGE_OWNER and w != EDGE_OWNER[et]:
                viol['S2 owner'].append(f"t={t} {ag(w)} wrote `{et}` {G.name(a)} -> {G.name(b)} (owner {ag(EDGE_OWNER[et])})")
            G.add_edge(a, b, et, w)
            if et == 'current':
                cur_ins.append((t, b)); room_changes.append((t, 'current', b, G.name(b), w))
                if len(G.currents()) > 1:
                    viol['S3 one-current'].append(f"t={t} {len(G.currents())} live `current` edges: {[G.name(c[1]) for c in G.currents()]}")
            if et == 'RT' and a in G.nodes and G.nodes[a]['type'] == 'robot' and b in G.nodes and G.nodes[b]['type'] == 'room':
                rt_new.append((t, b, G.name(b))); room_changes.append((t, 'RT', b, G.name(b), w))
            if et == 'match' and a in G.nodes and G.name(a).startswith('passage_'):
                faces = [k[1] for k in G.out[a] if k[2] == 'match' and G.live_edge(k)]
                if len(faces) == 2:
                    ra, rb = G.room_of(faces[0]), G.room_of(faces[1])
                    if ra is not None and ra == rb:
                        viol['S5 passage'].append(f"t={t} {G.name(a)} matches {G.name(faces[0])} and {G.name(faces[1])}, "
                                                  f"both in room {G.name(ra)}")
        elif ev == 'edge_delete':
            if et in EDGE_OWNER and w != EDGE_OWNER[et] and G.live_edge((a, b, et)):
                viol['S2 owner'].append(f"t={t} {ag(w)} deleted `{et}` {G.name(a)} -> {G.name(b)} (owner {ag(EDGE_OWNER[et])})")
            G.del_edge(a, b, et)
    # L2: room_concept's RT to a room ~> `current` on it within the bound
    for t, room, nm in rt_new:
        if not any(tc >= t - 2000 and tc <= t + bound_ms and rc == room for tc, rc in cur_ins):
            later = [tc for tc, rc in cur_ins if rc == room and tc > t]
            viol['L2 current'].append(f"t={t} robot RT -> {nm}: no `current` on it within {bound_ms/1000:.0f} s"
                                      + (f" (it came {(later[0]-t)/1000:.1f} s later)" if later else " (never)"))
    return viol, room_changes, G, info, extra


# ---------------------------------------------------------------- ground truth
def read_doors(world):
    txt = open(world).read()
    doors = {}
    for m in re.finditer(r'DEF (DOOR_\d+) Door \{', txt):
        body = txt[m.end():m.end() + 2000]
        get = lambda key: re.search(r'^\s*' + key + r'\s+([-\d.e\s]+)$', body, re.M)
        tr, rot, size = get('translation'), get('rotation'), get('size')
        if not (tr and rot): continue
        x, y = map(float, tr.group(1).split()[:2])
        ax = list(map(float, rot.group(1).split()[:4]))
        th = ax[3] * (1 if ax[2] >= 0 else -1)                       # rotation about +z
        width = float(size.group(1).split()[1]) if size else 1.0      # Door size = thickness, WIDTH, height
        doors[m.group(1)] = dict(c=(x, y), u=(-math.sin(th), math.cos(th)), n=(math.cos(th), math.sin(th)), w=width)
    return doors


def read_gt(path):
    rows = []
    with open(path) as f:
        hdr = f.readline().split('\t')
        for line in f:
            p = line.split('\t')
            if len(p) >= 3: rows.append((int(p[0]), float(p[1]), float(p[2])))
    return rows


def gt_crossings(gt, doors, margin=0.3):
    out = []
    for (t0, x0, y0), (t1, x1, y1) in zip(gt, gt[1:]):
        for name, d in doors.items():
            s0 = (x0 - d['c'][0]) * d['n'][0] + (y0 - d['c'][1]) * d['n'][1]
            s1 = (x1 - d['c'][0]) * d['n'][0] + (y1 - d['c'][1]) * d['n'][1]
            if s0 == s1 or (s0 > 0) == (s1 > 0): continue
            a = s0 / (s0 - s1)
            px, py = x0 + a * (x1 - x0), y0 + a * (y1 - y0)
            u = (px - d['c'][0]) * d['u'][0] + (py - d['c'][1]) * d['u'][1]
            if abs(u) <= 0.5 * d['w'] + margin:
                out.append((int(t0 + a * (t1 - t0)), name, 1 if s1 > 0 else -1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('run', help='run directory (holds graph_trace.csv and gt_pose.tsv)')
    ap.add_argument('--trace'); ap.add_argument('--gt'); ap.add_argument('--world', default=DEFAULT_WORLD)
    ap.add_argument('--bound', type=float, default=20.0, help='seconds allowed between a true crossing and a room fact')
    A = ap.parse_args()
    trace = A.trace or os.path.join(A.run, 'graph_trace.csv')
    gtp = A.gt or os.path.join(A.run, 'gt_pose.tsv')
    bound = int(A.bound * 1000)
    viol, changes, G, info, extra = check_trace(trace, bound)
    t_end = max(r['t'] for r in read_trace(trace))

    print(f"trace: {trace}")
    print("room-change facts:")
    for t, kind, room, nm, w in changes:
        print(f"  t={t}  {kind:7s} -> {nm:10s} by {ag(w)}")
    # C1/C2 churn: the same recall re-posted, the same proposal rejected, again and again (proto15: every 3 s)
    # Counted per STAY in a room (between room_concept's RT switches): ltsm re-posts a recall once per entry by design
    # (pingpong8: recall_room_1 once per entry into room_2, 3 entries = 3 posts, flagged as churn).
    sw = sorted(t for t, kind, room, nm, w in changes if kind == 'RT')
    stay = lambda t: sum(1 for x in sw if x <= t)
    rc_by = collections.Counter((n, stay(t)) for t, n in extra['recall_ins'])
    for (n, k), c in sorted(rc_by.items()):
        if c >= 3:
            ts = [t for t, m in extra['recall_ins'] if m == n and stay(t) == k]
            viol['C1 recall-churn'].append(f"{n} inserted {c} times in one room stay between t={ts[0]} and t={ts[-1]} "
                                           f"({(ts[-1] - ts[0]) / 1000:.0f} s)")
    rej = [x for x in extra['proposal'] if x[2] == 'rejected']
    by = collections.Counter(n for _, n, _, _ in rej)
    for n, c in sorted(by.items()):
        if c >= 3:
            ts = [t for t, m, _, _ in rej if m == n]
            viol['C2 proposal-churn'].append(f"{n} rejected {c} times between t={ts[0]} and t={ts[-1]}")
    if os.path.exists(gtp) and os.path.exists(A.world):
        doors = read_doors(A.world)
        gt = read_gt(gtp)
        xs = gt_crossings(gt, doors)
        print(f"ground truth: {len(gt)} poses {gt[0][0]}..{gt[-1][0]}; doors {', '.join(sorted(doors))}; "
              f"{len(xs)} true crossing(s)")
        for t, door, sgn in xs:
            hits = [(tc, kind, nm) for tc, kind, room, nm, w in changes if t - 2000 <= tc <= t + bound]
            if hits:
                print(f"  t={t} {door} ({'+' if sgn > 0 else '-'}n): room fact {hits[0][1]} -> {hits[0][2]} "
                      f"at {(hits[0][0] - t) / 1000:+.1f} s")
            else:
                nxt = [(tc, kind, nm) for tc, kind, room, nm, w in changes if tc > t + bound]
                tail = (f"next room fact {(nxt[0][0] - t) / 1000:.0f} s later ({nxt[0][1]} -> {nxt[0][2]})" if nxt
                        else f"NO room fact until the end of the trace ({(t_end - t) / 1000:.0f} s) or of gt "
                             f"({(gt[-1][0] - t) / 1000:.0f} s)")
                viol['L1 crossing'].append(f"t={t} the robot truly crossed {door}; no room-change fact within "
                                           f"{A.bound:.0f} s -- {tail}")
                print(f"  t={t} {door}: NO room fact within {A.bound:.0f} s")
        # S7: room_concept switched the robot into a NEW room node with no true crossing near it (inside the gt span)
        # ONE crossing justifies ONE switch: each true crossing is matched to the first unmatched RT switch in
        # [tx - 2 s, tx + bound]; a switch left unmatched is false (proto15: room_3 5 s after room_2, the robot in the doorway).
        rts = [(t, nm) for t, kind, room, nm, w in changes if kind == 'RT' and gt[0][0] <= t <= gt[-1][0]]
        used = set()
        for tx, door, _ in xs:
            for i, (t, nm) in enumerate(rts):
                if i not in used and tx - 2000 <= t <= tx + bound:
                    used.add(i); break
        for i, (t, nm) in enumerate(rts):
            if i not in used:
                viol['S7 false-switch'].append(f"t={t} room_concept moved the robot into {nm}; no true door crossing is left "
                                               f"to explain it (each crossing explains one switch)")
        # S8: a mission leg reported `completed` with no true crossing since it went in_progress
        start = None
        for t, st in extra['mission']:
            if st == 'in_progress': start = t
            elif st == 'completed' and start is not None:
                if not any(start <= tx <= t for tx, _, _ in xs):
                    viol['S8 leg-without-crossing'].append(f"leg in_progress t={start} -> completed t={t}: the robot crossed "
                                                           f"no door in between (ground truth)")
                start = None
    else:
        print(f"(no ground truth: {gtp} / {A.world} missing -- L1, S7, S8 skipped)")

    for line in info:
        print(f"note: {line}")
    print()
    names = ['S1 exec-door', 'S2 owner', 'S3 one-current', 'S4 names', 'S5 passage', 'S6 recall-live', 'S7 false-switch',
             'S8 leg-without-crossing', 'L1 crossing', 'L2 current', 'C1 recall-churn', 'C2 proposal-churn']
    for k in names:
        v = viol.get(k, [])
        print(f"{k:15s} {'OK' if not v else f'{len(v)} VIOLATION(S)'}")
        for line in v[:12]:
            print(f"    {line}")
        if len(v) > 12: print(f"    ... {len(v) - 12} more")
    return 1 if any(viol.values()) else 0


if __name__ == '__main__':
    sys.exit(main())
