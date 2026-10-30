#!/usr/bin/env python3
"""protocol_mock.py -- seconds-fast simulation of the ROOM-TRANSITION PROTOCOL under adversarial scheduling.

CAVEAT (read first): this models the protocol AS DESIGNED (docs/superpowers/specs/2026-10-29-room-transition-protocol-
design.md), NOT the agents' C++.  A pass here says the DESIGN is consistent under adversarial ordering; bugs inside an
agent's own code are pinned in that agent's C++ tests.  Every place where the spec is silent, the mock had to INVENT a rule
(marked "SPEC GAP" below); those are the design's open questions, listed in protocol_mock_report.md.

Model
  world      rooms R1,R2,R3; holes DOOR_1 (R1-R2) DOOR_0 (R1-R3) DOOR_2 (R3-R2).  Ground truth (true room, hole being
             passed) lives only in World; agents see the graph plus their own sensors (see each agent).
  graph      keyed LWW store (node base / node attribute / edge keys).  Every write records its writer and is checked
             against the owners table.  Each agent reads its OWN VIEW; a write reaches the other views after a random
             delay (DDS: batches/topics are unordered), the author sees it at once.  LWW per key by sequence number.
  agents     room_concept (RC), door_concept (DC), ltsm_agent (LT), controller (CT): one class each, step() reads the view
             and writes only what the owners table allows (case 1 / case 2 of the spec).
  scheduler  per tick the agents step in random order; per seed: random visibility delay, random "drop" delay, random agent
             stall windows.  --dmax/--drop/--stall/--lag set the adversary; --seed N --trace prints one schedule.
  tour       6 legs (recognition_tour.sh): new->R2, room_1, new->R3, new->R2 (recognised), room_3, room_1.
Invariants (Violation.inv): 1 ownership  2 act-on-fact (stale absence, transform provenance)  3 passage joins two rooms
  4 faces (one room, right hole, one per hole, parent exists)  5 controller room == true room / right destination
  6 recognised room renamed in place, never destroyed/duplicated  7 no mission waits past a bound (deadlock/livelock)
  8 names unique among live rooms / live faces   9 end-of-tour memory consistent.
"""
import argparse, collections, random, sys

HOLES = {'DOOR_1': ('R1', 'R2'), 'DOOR_0': ('R1', 'R3'), 'DOOR_2': ('R3', 'R2')}
DIST = {'DOOR_1': 3.0, 'DOOR_0': 5.0, 'DOOR_2': 4.0}          # metres from the robot (pick_door tie-break)
T_FRAME = {'R1': 0.0, 'R2': 1.0, 'R3': 2.0}                   # the true transform room <- door frame; recall prior = +0.05
LEGS = [('new', 'R2'), ('room_1', 'R1'), ('new', 'R3'), ('new', 'R2'), ('room_3', 'R3'), ('room_1', 'R1')]
RC, DC, LT, CT = 'room_concept', 'door_concept', 'ltsm_agent', 'controller'
TYPE_OWNER = {'room': RC, 'door': DC, 'passage': LT, 'recall': LT, 'proposal': LT, 'archive': LT}
EDGE_OWNER = {'current': RC, 'proto': RC, 'match': LT}
SHARED_ATTR = {('proposal', 'verdict'): RC}                   # SPEC GAP: spec says RC writes "rejected" ON the proposal


def holes_of(r): return [h for h, (a, b) in HOLES.items() if r in (a, b)]
def other_room(h, r): a, b = HOLES[h]; return b if r == a else a


class Violation(Exception):
    def __init__(s, inv, msg): super().__init__(msg); s.inv = inv; s.msg = msg


# ------------------------------------------------------------------------------------------------ graph
class View:
    def __init__(s):
        s.seq = {}; s.nodes = {}; s.attrs = collections.defaultdict(dict)
        s.edges = collections.defaultdict(dict); s.by_type = collections.defaultdict(set); s.kids = collections.defaultdict(set)
        s.pending = []

    def apply(s, key, seq, val):
        if s.seq.get(key, -1) >= seq: return
        s.seq[key] = seq
        if key[0] == 'n':
            old = s.nodes.pop(key[1], None)
            if old: s.by_type[old['type']].discard(key[1]); s.kids[old['parent']].discard(key[1])
            if val is not None:
                s.nodes[key[1]] = val; s.by_type[val['type']].add(key[1]); s.kids[val['parent']].add(key[1])
            else: s.attrs.pop(key[1], None)
        elif key[0] == 'a':
            if key[1] not in s.nodes and val is None: return
            if val is None: s.attrs[key[1]].pop(key[2], None)
            else: s.attrs[key[1]][key[2]] = val
        else:
            if val: s.edges[key[1]][(key[2], key[3])] = seq
            else: s.edges[key[1]].pop((key[2], key[3]), None)

    def attr(s, i, k, d=None): return s.attrs[i].get(k, d) if i in s.nodes else d
    def name(s, i): return s.attr(i, 'name')
    def kids_of(s, i, typ): return [k for k in sorted(s.kids[i]) if s.nodes[k]['type'] == typ]
    def current(s):
        e = s.edges['current']
        return max(e.items(), key=lambda kv: kv[1])[0][1] if e else None
    def has_edge(s, kind, a, b): return (a, b) in s.edges[kind]
    def proto(s, i): return s.has_edge('proto', i, i)


class Store:
    def __init__(s, rng, cfg, agents):
        s.rng = rng; s.cfg = cfg; s.seq = 0; s.nid = 0; s.auth = View(); s.views = {a: View() for a in agents}
        s.log = []; s.last_write = 0; s.t = 0; s.T_reads = []; s.lag = cfg.lag_map

    def new_id(s): s.nid += 1; return s.nid

    def check_owner(s, who, key, val):
        a = s.auth
        if key[0] == 'n':
            node = a.nodes.get(key[1])
            if node is None:
                if val is None: return
                if TYPE_OWNER[val['type']] != who or val['owner'] != who:
                    raise Violation(1, f"{who} created {val['type']} node {key[1]} owned by {TYPE_OWNER[val['type']]}")
            elif node['owner'] != who:
                raise Violation(1, f"{who} {'deleted' if val is None else 'rewrote'} {node['type']} node {key[1]} "
                                   f"({a.name(key[1])}) owned by {node['owner']}")
        elif key[0] == 'a':
            node = a.nodes.get(key[1])
            if node is None: return
            if node['owner'] != who and SHARED_ATTR.get((node['type'], key[2])) != who:
                raise Violation(1, f"{who} wrote attr {key[2]} of {node['type']} {key[1]} owned by {node['owner']}")
        else:
            if EDGE_OWNER[key[1]] != who: raise Violation(1, f"{who} wrote {key[1]} edge owned by {EDGE_OWNER[key[1]]}")

    def write(s, who, ops, t, note=''):
        for key, val in ops: s.check_owner(who, key, val)
        for key, val in ops:
            s.seq += 1; sq = s.seq
            s.auth.apply(key, sq, val); s.views[who].apply(key, sq, val)
            s.log.append(f"t={t:<4}{who[:6]:<7}{s._fmt(key, val)}")
        # one visibility delay per reader per batch (a batch is one atomic node update); different batches are unordered
        first = s.seq - len(ops) + 1
        for r, v in s.views.items():
            if r == who: continue
            d = s.rng.randint(0, s.cfg.run_dmax) + s.lag.get(who, 0)
            if s.rng.random() < s.cfg.pdrop: d += s.rng.randint(1, s.cfg.drop)
            for i, (key, val) in enumerate(ops): v.pending.append((t + d, first + i, key, val))
        s.last_write = t

    def _fmt(s, key, val):
        if key[0] == 'n':
            return f"DEL n{key[1]}" if val is None else f"NEW n{key[1]} {val['type']} parent={val['parent']} truth={val['truth']}"
        if key[0] == 'a': return f"SET n{key[1]}.{key[2]} = {val}"
        return f"{'ADD' if val else 'DEL'} edge {key[1]} {key[2]}->{key[3]}"

    def deliver(s, t):
        for v in s.views.values():
            due = [p for p in v.pending if p[0] <= t]
            if not due: continue
            v.pending = [p for p in v.pending if p[0] > t]
            s.rng.shuffle(due)
            for _, sq, key, val in due: v.apply(key, sq, val)


def node_ops(nid, typ, parent, owner, truth, attrs):
    return [(('n', nid), {'type': typ, 'parent': parent, 'owner': owner, 'truth': truth})] + \
           [(('a', nid, k), v) for k, v in attrs.items()]


# ------------------------------------------------------------------------------------------------ world
class World:
    def __init__(s, rng, cfg):
        s.rng = rng; s.cfg = cfg; s.true_room = 'R1'; s.epoch = 0; s.passed_hole = None; s.crossing = None
        s.enter_tick = 0; s.inside_ready = 0; s.detect = {h: rng.randint(1, cfg.detect) for h in holes_of('R1')}

    def tick(s, t):
        c = s.crossing
        if c and t >= c['done']:
            s.true_room = c['to']; s.epoch += 1; s.passed_hole = c['hole']; s.enter_tick = t; s.crossing = None
            s.inside_ready = t + s.rng.randint(1, 3)
            s.detect = {h: s.rng.randint(1, s.cfg.detect) for h in holes_of(s.true_room)}

    def start_cross(s, hole, t):
        assert hole in holes_of(s.true_room) and s.crossing is None
        s.crossing = {'hole': hole, 'to': other_room(hole, s.true_room), 'done': t + s.cfg.drive}

    def sensed(s, t):   # holes door_concept's sensors resolve now (the entry hole at once, others after a detection delay)
        return [h for h in holes_of(s.true_room)
                if (h == s.passed_hole and s.epoch > 0) or t - s.enter_tick >= s.detect[h]]


class Agent:
    def __init__(s, name, store, w, rng, cfg):
        s.name = name; s.S = store; s.v = store.views[name]; s.w = w; s.rng = rng; s.cfg = cfg; s.init()
    def init(s): pass
    def put(s, ops, t): s.S.write(s.name, ops, t)
    def note(s, t, msg): s.S.log.append(f"t={t:<4}{s.name[:6]:<7}# {msg}")


# ------------------------------------------------------------------------------------------------ room_concept
class RoomConcept(Agent):
    def init(s): s.hwm = 0; s.handled = 0; s.started = False; s.created_prov = {}; s.renames = []

    def new_room(s, t, name, truth, attrs, old=None, prov=False):
        i = s.S.new_id(); ops = node_ops(i, 'room', None, RC, truth, dict(attrs, name=name, entered_epoch=s.w.epoch))
        if old is not None: ops.append((('e', 'current', 'root', old), None))
        ops.append((('e', 'current', 'root', i), True))
        if prov: ops.append((('e', 'proto', i, i), True))
        s.put(ops, t); return i

    def fresh_name(s): s.hwm += 1; return f"room_{s.hwm}"

    def step(s, t):
        v, w = s.v, s.w
        if not s.started:
            s.started = True
            s.new_room(t, s.fresh_name(), 'R1', dict(state='normal', outline='R1', frame_T=0.0)); return
        if w.epoch > s.handled and t >= w.inside_ready: s.switch(t)
        cur = v.current()
        if cur in v.nodes and v.proto(cur):
            e = v.attr(cur, 'est_steps', 0)
            if e < 12:
                ops = [(('a', cur, 'est_steps'), e + 1)]
                if e + 1 >= 3 and v.attr(cur, 'outline') is None:
                    ops += [(('a', cur, 'outline'), v.nodes[cur]['truth']), (('a', cur, 'frame_T'), T_FRAME[v.nodes[cur]['truth']])]
                s.put(ops, t)
            for p in v.kids_of(cur, 'proposal'):
                if v.attr(p, 'verdict') is not None: continue
                if v.attr(p, 'target_outline') == v.attr(cur, 'outline'):      # its own scans agree with the proposal
                    tt = T_FRAME[v.nodes[p]['truth']]
                    s.put([(('a', cur, 'name'), v.attr(p, 'target_name')), (('a', cur, 'outline'), v.attr(p, 'target_outline')),
                           (('a', cur, 'frame_T'), tt), (('a', cur, 'accepted_T'), tt), (('a', cur, 'mkey'), v.attr(p, 'target_mkey')),
                           (('a', cur, 'state'), 'adopted'), (('e', 'proto', cur, cur), None), (('a', p, 'verdict'), 'accepted')], t)
                    s.renames.append((cur, v.attr(p, 'target_name')))
                else: s.put([(('a', p, 'verdict'), 'rejected')], t)
        arch = {v.attr(a, 'room_id') for a in v.by_type['archive']}
        for r in sorted(v.by_type['room']):                                  # delete a left room only once archived AND its faces are gone
            if r != cur and r in arch and not v.kids_of(r, 'door'):
                ops = [(('n', r), None)]
                if v.proto(r): ops.append((('e', 'proto', r, r), None))
                s.put(ops, t)

    def switch(s, t):
        v, w = s.v, s.w
        old = v.current(); hole = w.passed_hole
        face = next((f for f in v.kids_of(old, 'door') if v.attr(f, 'hole') == hole), None) if old in v.nodes else None
        recall = next(iter(v.kids_of(face, 'recall')), None) if face is not None else None
        s.handled = w.epoch
        if recall is None:                       # CASE 2: nothing known behind this door -> provisional room
            a = s.S.auth
            for f in a.kids_of(old, 'door'):
                if a.attr(f, 'hole') == hole and a.kids_of(f, 'recall'):
                    raise Violation(2, f"room_concept took CASE 2 (provisional room) at {hole} but ltsm had ALREADY written "
                                       f"recall under face n{f}; not yet visible to room_concept (acted on absence of a fact)")
            nm = s.fresh_name()
            i = s.new_room(t, nm, w.true_room, dict(state='provisional', est_steps=0), old=old, prov=True)
            s.created_prov[i] = w.true_room; s.note(t, f"CASE2 provisional {nm} n{i} truth={w.true_room}")
        else:                                    # CASE 1: instantiate the remembered room from the recall
            tr = s.v.nodes[recall]['truth']
            if tr != w.true_room: raise Violation(5, f"recall n{recall} names {tr} but the robot is truly in {w.true_room}")
            nm = v.attr(recall, 'room_name'); a = s.hwm
            i = s.new_room(t, nm, tr, dict(state='recalled', outline=v.attr(recall, 'outline'), frame_T=v.attr(recall, 'frame_T'),
                                             mkey=v.attr(recall, 'mkey')), old=old)
            s.note(t, f"CASE1 recalled {nm} n{i} from recall n{recall}")


# ------------------------------------------------------------------------------------------------ door_concept
class DoorConcept(Agent):
    def init(s): s.mem = {}; s.last = None; s.last_info = None; s.used = set(); s.nextn = 0

    def fresh(s, v):
        live = {v.name(f) for f in v.by_type['door']}
        while True:
            s.nextn += 1; n = f"door_{s.nextn}"
            if n not in live and n not in s.used: s.used.add(n); return n

    def tfor(s, v, cur, state):
        """The transform a restored face is expressed with; provenance recorded for the invariant-2 audit."""
        if state == 'adopted':
            if s.cfg.fault == 'read_prior_not_accepted':
                p = next(iter(v.kids_of(cur, 'proposal')), None)
                if p is not None:
                    s.S.T_reads.append((s.name, p, 'T_prior', s.S.t)); return v.attr(p, 'T_prior')
            s.S.T_reads.append((s.name, cur, 'accepted_T', s.S.t)); return v.attr(cur, 'accepted_T')
        s.S.T_reads.append((s.name, cur, 'frame_T', s.S.t)); return v.attr(cur, 'frame_T')

    def step(s, t):
        v, w = s.v, s.w
        arch = {v.attr(a, 'room_id') for a in v.by_type['archive']}
        for f in sorted(v.by_type['door']):
            p = v.nodes[f]['parent']
            if p in arch or p not in v.nodes: s.put([(('n', f), None)], t)
        cur = v.current()
        if cur != s.last:                                           # promotion: a provisional room we leave is remembered under its outline
            if s.last_info and s.last_info[0] == 'provisional' and s.last_info[1]:
                s.mem.setdefault(s.last_info[1], {}).update(s.mem.pop(('n', s.last), {}))
            s.last = cur
        if cur not in v.nodes or v.attr(cur, 'entered_epoch') != w.epoch: return    # robot not (yet) inside this room
        state = v.attr(cur, 'state'); outline = v.attr(cur, 'outline'); s.last_info = (state, outline)
        faces = {v.attr(f, 'hole'): f for f in v.kids_of(cur, 'door')}
        key = ('n', cur) if state == 'provisional' else outline
        mem = s.mem.setdefault(key, {}); want = []
        if state == 'recalled': want = list(mem)                    # restore remembered doors (incl. the entry face)
        elif state == 'provisional':
            if w.epoch > 0 and w.passed_hole is not None: want = [w.passed_hole]       # entry mirror
        elif state == 'adopted' and v.attr(cur, 'accepted_T') is not None:
            s.mem.pop(('n', cur), None)
            for h, f in faces.items():                               # merge by place: keep P's face, give it the remembered name
                if h in mem and v.name(f) != mem[h] and mem[h] not in {v.name(x) for x in v.by_type['door']}:
                    s.put([(('a', f, 'name'), mem[h])], t)
                tu = s.tfor(v, cur, state)
                if v.attr(f, 'T_used') != tu: s.put([(('a', f, 'T_used'), tu)], t)
            want = list(mem)
        want += [h for h in w.sensed(t) if h not in want]
        entry = w.passed_hole if w.epoch > 0 else None
        for h in want:
            if h in faces: continue
            nm = mem.get(h); nm = nm if nm and nm not in {v.name(x) for x in v.by_type['door']} else s.fresh(v)
            attrs = dict(name=nm, hole=h)
            if state in ('recalled', 'adopted'): attrs['T_used'] = s.tfor(v, cur, state)
            if s.cfg.entry_mark and h == entry: attrs['entry'] = 1
            fid = s.S.new_id(); s.put(node_ops(fid, 'door', cur, DC, None, attrs), t); faces[h] = fid
        for h, f in faces.items(): mem[h] = v.name(f)


# ------------------------------------------------------------------------------------------------ ltsm_agent
class Ltsm(Agent):
    def init(s):
        s.rooms = {}; s.passages = {}; s.node_key = {}; s.cur = None; s.pending = None; s.nk = 0; s.ev = {}; s.carried = set()
        s.proposed = {}; s.pass_node = {}

    def newkey(s): s.nk += 1; return f"m{s.nk}"
    def key_for(s, r):
        k = s.v.attr(r, 'mkey')
        return k if k is not None else s.node_key.setdefault(r, s.newkey())

    def carry_over(s, r, t):                                          # step 9: the provisional room's passage joins the recalled room
        v = s.v
        if r in s.carried or v.attr(r, 'state') != 'adopted' or v.attr(r, 'mkey') is None: return
        mp = s.node_key.get(r); x = v.attr(r, 'mkey'); s.carried.add(r); s.node_key[r] = x
        if mp is None or mp == x: return
        for pid, p in s.passages.items():
            for e in p['ends']:
                if e[0] == mp: e[0] = x
            s.put([(('a', s.pass_node[pid], 'ends'), tuple(tuple(e) for e in p['ends']))], t)
        s.note(t, f"carry-over {mp}->{x}")

    def snap(s, r):
        v = s.v
        return dict(id=r, name=v.name(r), outline=v.attr(r, 'outline'), frame_T=v.attr(r, 'frame_T'), truth=v.nodes[r]['truth'])

    def step(s, t):
        v, w = s.v, s.w
        for kind in ('recall', 'proposal'):                          # own nodes whose parent is gone
            for n in sorted(v.by_type[kind]):
                if v.nodes[n]['parent'] not in v.nodes: s.put([(('n', n), None)], t)
        for (pn, f), _ in list(v.edges['match'].items()):
            if f not in v.nodes: s.put([(('e', 'match', pn, f), None)], t)
        cur = v.current()
        if s.pending and cur != s.pending['new']: s.note(t, "transition abandoned (missed a room)"); s.cur = s.pending['new']; s.pending = None
        if s.pending is None and cur in v.nodes and cur != s.cur:
            if s.cur is None: s.cur = cur; s.key_for(cur)
            else:
                old = s.cur
                if old in v.nodes:
                    s.carry_over(old, t)
                    sn = s.snap(old); sn['key'] = s.key_for(old)
                    s.pending = dict(old=sn, new=cur, hole=w.passed_hole); s.cur = cur
                else: s.cur = cur
        if s.pending: s.finish_transition(t)
        cur = v.current()
        if cur in v.nodes:
            s.carry_over(cur, t); s.reconcile(cur, t); s.recognise(cur, t)

    def finish_transition(s, t):
        v = s.v; p = s.pending; new = p['new']; hole = p['hole']
        if new not in v.nodes: return
        if not any(v.attr(f, 'hole') == hole for f in v.kids_of(new, 'door')): return      # wait for the entry face (fact 4)
        o = p['old']; nk = s.key_for(new)
        s.rooms[o['key']] = dict(name=o['name'], outline=o['outline'], frame_T=o['frame_T'], truth=o['truth'])
        pid = next((i for i, q in s.passages.items() if any(e[0] == o['key'] and e[1] == hole for e in q['ends'])), None)
        if pid is None:
            pid = len(s.passages); s.passages[pid] = dict(ends=[[o['key'], hole], [nk, hole]], crossings=1)
            nid = s.S.new_id(); s.pass_node[pid] = nid
            s.put(node_ops(nid, 'passage', None, LT, None, dict(name=f"passage_{pid}", ends=tuple(tuple(e) for e in s.passages[pid]['ends']),
                                                                   crossings=1)), t)
        else:
            s.passages[pid]['crossings'] += 1
            s.put([(('a', s.pass_node[pid], 'crossings'), s.passages[pid]['crossings'])], t)
        s.reconcile(new, t)
        aid = s.S.new_id(); s.put(node_ops(aid, 'archive', None, LT, None, dict(name=f"archive_{o['id']}", room_id=o['id'])), t)
        s.note(t, f"archived {o['name']} key={o['key']} via {hole}; passage_{pid}"); s.pending = None

    def reconcile(s, cur, t):
        v = s.v; k = s.key_for(cur)
        for f in v.kids_of(cur, 'door'):
            h = v.attr(f, 'hole')
            for pid, q in s.passages.items():
                ends = q['ends']
                for i in (0, 1):
                    if ends[i][0] == k and ends[i][1] == h:
                        if not v.has_edge('match', s.pass_node[pid], f): s.put([(('e', 'match', s.pass_node[pid], f), True)], t)
                        ok = ends[1 - i][0]
                        if ok in s.rooms and not any(v.attr(c, 'mkey') == ok for c in v.kids_of(f, 'recall')):
                            m = s.rooms[ok]
                            s.put(node_ops(s.S.new_id(), 'recall', f, LT, m['truth'],
                                           dict(name=f"recall_{m['name']}", room_name=m['name'], outline=m['outline'],
                                                frame_T=m['frame_T'], mkey=ok)), t)
        if s.cfg.fault == 'pair_same_room' and v.proto(cur):
            fs = [(v.attr(f, 'hole'), f) for f in v.kids_of(cur, 'door')]
            matched = [h for h, f in fs if any(v.has_edge('match', s.pass_node[pid], f) for pid in s.passages)]
            free = [h for h, f in fs if h not in matched]
            if matched and free and not any(e[0] == k and e[1] == free[0] and e2[0] == k for q in s.passages.values()
                                            for e, e2 in (q['ends'], q['ends'][::-1])):
                pid = len(s.passages); s.passages[pid] = dict(ends=[[k, matched[0]], [k, free[0]]], crossings=0)
                nid = s.S.new_id(); s.pass_node[pid] = nid
                s.put(node_ops(nid, 'passage', None, LT, None, dict(name=f"passage_{pid}", ends=((k, matched[0]), (k, free[0])), crossings=0)), t)

    def recognise(s, cur, t):
        v = s.v
        if not v.proto(cur): return
        props = v.kids_of(cur, 'proposal')
        if any(v.attr(p, 'verdict') != 'rejected' for p in props): return
        # evidence = ltsm's OWN observation steps of this room (its filtered metric evidence accrues per scan, not per RC write)
        o = v.attr(cur, 'outline')
        if o is None: return
        if props: s.ev[cur] = 0                                      # rebased at a rejection
        s.ev[cur] = s.ev.get(cur, 0) + 1
        if s.ev[cur] < s.cfg.evidence: return
        k = s.key_for(cur)
        cand = [kk for kk, m in s.rooms.items() if m['outline'] == o and kk != k]
        if not cand: return
        m = s.rooms[cand[0]]
        s.put(node_ops(s.S.new_id(), 'proposal', cur, LT, m['truth'],
                       dict(name=f"proposal_{m['name']}", target_mkey=cand[0], target_name=m['name'], target_outline=o,
                            T_prior=m['frame_T'] + 0.05)), t)
        s.note(t, f"PROPOSAL: n{cur} is {m['name']}")
        if s.cfg.fault == 'ltsm_deletes_provisional': s.put([(('n', cur), None)], t)       # the old design's swap


# ------------------------------------------------------------------------------------------------ controller
class Controller(Agent):
    def init(s): s.leg = 0; s.phase = 'start'; s.timer = 0; s.start_id = None; s.since = 0; s.done = False; s.legs_done = []

    def facts(s, cur):
        v = s.v; out = []
        for f in v.kids_of(cur, 'door'):
            rec = {v.attr(c, 'room_name') for c in v.kids_of(f, 'recall')}
            m = any(f == b for (a, b) in v.edges['match'])
            out.append(dict(name=v.name(f), id=f, recalls=rec, match=m, entry=bool(v.attr(f, 'entry')), dist=DIST[v.attr(f, 'hole')]))
        return out

    def pick(s, rel, doors):
        best = None
        for d in doors:
            if rel == 'new':
                if d['recalls'] or d['match'] or (s.cfg.entry_mark and d['entry']): continue
                if best is None or (d['dist'], d['name']) < (best['dist'], best['name']): best = d
            elif rel in d['recalls']:
                if best is None or d['name'] < best['name']: best = d
        return best

    def step(s, t):
        if s.leg >= len(LEGS): s.done = True; return
        v, w = s.v, s.w; rel, expect = LEGS[s.leg]; cur = v.current()
        if s.phase == 'start':
            if cur in v.nodes: s.phase = 'explore'; s.timer = s.cfg.explore; s.start_id = cur
        elif s.phase == 'explore':
            s.timer -= 1
            if s.timer <= 0: s.phase = 'hold'; s.since = t
        elif s.phase == 'hold':
            d = s.pick(rel, s.facts(cur)) if cur in v.nodes else None
            if d is None:
                s.wait_desc = f"leg {s.leg + 1} cross {rel}: no door resolves in {v.name(cur)}"
                if t - s.since > s.cfg.hold: s.stuck(t)
                return
            a = s.S.auth
            if cur not in a.nodes or a.nodes[cur]['truth'] != w.true_room:
                raise Violation(5, f"controller (room {v.name(cur)}) drives while the robot is truly in {w.true_room}")
            if d['id'] not in a.nodes: raise Violation(2, f"controller drives at door n{d['id']} that no longer exists")
            if rel == 'new':
                af = d['id']
                if a.kids_of(af, 'recall') or any(f == af for (_, f) in a.edges['match']):
                    raise Violation(2, f"controller took {d['name']} as 'new' but ltsm had ALREADY written its recall/match "
                                       f"(not yet visible to the controller): acted on the absence of a fact (try30)")
            hole = a.attr(d['id'], 'hole')
            if hole not in holes_of(w.true_room): raise Violation(5, f"door {d['name']} ({hole}) is not in the true room {w.true_room}")
            w.start_cross(hole, t); s.phase = 'drive'
        elif s.phase == 'drive':
            if w.crossing is None: s.phase = 'switch'; s.since = t
        elif s.phase == 'switch':
            if cur in v.nodes and cur != s.start_id:
                a = s.S.auth
                if a.nodes[cur]['truth'] != w.true_room:
                    raise Violation(5, f"leg {s.leg + 1} done: controller's room {v.name(cur)} is {a.nodes[cur]['truth']}, robot truly in {w.true_room}")
                if w.true_room != expect: raise Violation(5, f"leg {s.leg + 1} (cross {rel}) landed in {w.true_room}, expected {expect}")
                s.legs_done.append((s.leg + 1, t, v.name(cur), cur)); s.note(t, f"LEG {s.leg + 1} done in {v.name(cur)} n{cur}")
                s.leg += 1; s.phase = 'start'
            else:
                s.wait_desc = f"leg {s.leg + 1} waiting for the room switch (fact 3)"
                if t - s.since > s.cfg.hold: s.stuck(t)

    def stuck(s, t):
        quiet = t - s.S.last_write
        raise Violation(7, f"{s.wait_desc} for > {s.cfg.hold} ticks; last graph write {quiet} ticks ago "
                           f"({'DEADLOCK: nothing is coming' if quiet > 20 else 'LIVELOCK: facts keep changing'})")


# ------------------------------------------------------------------------------------------------ checks
def check_tick(S, w, agents, prov_ids):
    a = S.auth; t = S.t
    for who, src, attr, tt in S.T_reads:
        if attr not in ('accepted_T', 'frame_T'):
            raise Violation(2, f"{who} used transform '{attr}' of n{src} (a PROPOSAL prior) instead of the accepted/remembered one")
    S.T_reads.clear()
    for pid in a.by_type['passage']:
        ends = a.attr(pid, 'ends')
        if ends and ends[0][0] == ends[1][0]:
            raise Violation(3, f"passage n{pid} joins two doors of the SAME room (key {ends[0][0]}): {ends}")
        tgt = [f for (p, f) in a.edges['match'] if p == pid and f in a.nodes]
        rooms = {a.nodes[f]['parent'] for f in tgt}
        if len(tgt) != len(rooms) or len(tgt) > 2: raise Violation(3, f"passage n{pid} matches faces {tgt} in rooms {rooms}")
    seen = set()
    for f in a.by_type['door']:
        p = a.nodes[f]['parent']
        if p not in a.nodes: raise Violation(4, f"door face n{f} ({a.name(f)}) has no parent room (n{p} gone)")
        h = a.attr(f, 'hole')
        if h not in holes_of(a.nodes[p]['truth']): raise Violation(4, f"face n{f} {a.name(f)} hole {h} is not a hole of its room {a.name(p)} ({a.nodes[p]['truth']})")
        if (p, h) in seen: raise Violation(4, f"two faces for hole {h} in room {a.name(p)}")
        seen.add((p, h))
    for typ in ('room', 'door'):
        names = collections.Counter(a.name(n) for n in a.by_type[typ])
        dup = [n for n, c in names.items() if c > 1]
        if dup: raise Violation(8, f"duplicate live {typ} names {dup}")
    for i in list(prov_ids):
        if i not in a.nodes:
            if any(a.attr(x, 'room_id') == i for x in a.by_type['archive']): prov_ids.discard(i)
            else: raise Violation(6, f"provisional room n{i} destroyed (never renamed in place / left without an archive)")


def final_checks(S, agents):
    lt = agents[LT]; a = S.auth
    outs = collections.Counter(m['outline'] for m in lt.rooms.values())
    if any(c > 1 for c in outs.values()): raise Violation(6, f"ltsm memory stores a room twice (never recognised): {dict(outs)}")
    rc = agents[RC]
    if not any(nm == 'room_2' for _, nm in rc.renames): raise Violation(6, "the revisited room_2 was never adopted in place")
    for pid, q in lt.passages.items():
        if q['ends'][0][0] == q['ends'][1][0]: raise Violation(9, f"memory passage_{pid} loops on one room")
    if len(lt.passages) != 3: raise Violation(9, f"memory holds {len(lt.passages)} passages, expected 3: {[q['ends'] for q in lt.passages.values()]}")


# ------------------------------------------------------------------------------------------------ run
def run_one(seed, cfg):
    rng = random.Random(seed)
    cfg = argparse.Namespace(**vars(cfg)); cfg.run_dmax = rng.randint(0, cfg.dmax)
    names = [RC, DC, LT, CT]
    S = Store(rng, cfg, names); w = World(rng, cfg)
    ag = {RC: RoomConcept(RC, S, w, rng, cfg), DC: DoorConcept(DC, S, w, rng, cfg), LT: Ltsm(LT, S, w, rng, cfg), CT: Controller(CT, S, w, rng, cfg)}
    stalls = {n: (rng.randint(5, 200), rng.randint(1, cfg.stall)) for n in names if cfg.stall and rng.random() < 0.5}
    rc = ag[RC]; prov = set(); t = 0
    try:
        while not ag[CT].done or t < ag[CT].legs_done[-1][1] + 40:
            t += 1; S.t = t
            if t > 4000: raise Violation(7, f"tour not finished after 4000 ticks (legs done {len(ag[CT].legs_done)})")
            w.tick(t); S.deliver(t)
            order = names[:]; rng.shuffle(order)
            for n in order:
                st = stalls.get(n)
                if st and st[0] <= t < st[0] + st[1]: continue
                ag[n].step(t)
                prov |= set(rc.created_prov)
            check_tick(S, w, ag, prov)
        final_checks(S, ag)
        return None, S.log
    except Violation as e:
        S.log.append(f"t={t:<4}*** VIOLATION inv {e.inv}: {e.msg}")
        return e, S.log


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--seeds', type=int, default=500); ap.add_argument('--seed0', type=int, default=0)
    ap.add_argument('--seed', type=int, help='run and print the full trace of this one seed')
    ap.add_argument('--dmax', type=int, default=2, help='max per-write visibility delay (ticks); each seed draws 0..dmax')
    ap.add_argument('--drop', type=int, default=4, help='extra delay of a "dropped" write (prob 0.05)'); ap.add_argument('--pdrop', type=float, default=0.05)
    ap.add_argument('--stall', type=int, default=3, help='max agent stall length (ticks)')
    ap.add_argument('--lag', default='', help='AGENT=K[,AGENT=K]: extra latency of every write by AGENT')
    ap.add_argument('--explore', type=int, default=24); ap.add_argument('--drive', type=int, default=6)
    ap.add_argument('--detect', type=int, default=6); ap.add_argument('--evidence', type=int, default=6)
    ap.add_argument('--hold', type=int, default=120)
    ap.add_argument('--fault', choices=['pair_same_room', 'read_prior_not_accepted', 'ltsm_deletes_provisional'])
    ap.add_argument('--entry-mark', dest='entry_mark', action='store_true', help='mitigation: door_concept marks the entry face; "cross new" skips it')
    ap.add_argument('--stress', action='store_true', help='dmax 10, drop 12, stall 12')
    ap.add_argument('--tail', type=int, default=90, help='trace lines printed for the first failure')
    cfg = ap.parse_args()
    if cfg.stress: cfg.dmax, cfg.drop, cfg.stall = 10, 12, 12
    cfg.lag_map = {{'room': RC, 'door': DC, 'ltsm': LT, 'controller': CT}.get(k, k): int(v) for k, v in (x.split('=') for x in cfg.lag.split(',') if x)}
    if cfg.seed is not None:
        e, log = run_one(cfg.seed, cfg); print('\n'.join(log)); print('RESULT:', 'PASS' if e is None else f'FAIL inv {e.inv}'); return 0 if e is None else 1
    fails = collections.Counter(); first = None; ok = 0
    for sd in range(cfg.seed0, cfg.seed0 + cfg.seeds):
        e, log = run_one(sd, cfg)
        if e is None: ok += 1
        else:
            fails[e.inv] += 1
            if first is None: first = (sd, e, log)
    print(f"fault={cfg.fault} entry_mark={cfg.entry_mark} dmax={cfg.dmax} stall={cfg.stall} explore={cfg.explore}: "
          f"{ok}/{cfg.seeds} pass, {cfg.seeds - ok} fail; failing invariants {dict(sorted(fails.items()))}")
    if first:
        sd, e, log = first
        print(f"-- first failure: seed {sd}, invariant {e.inv}: {e.msg}\n-- trace (last {cfg.tail} lines)")
        print('\n'.join(log[-cfg.tail:]))
    return 0 if (ok == cfg.seeds) == (cfg.fault is None) else 1


if __name__ == '__main__':
    sys.exit(main())
