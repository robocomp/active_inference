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
import argparse, collections, math, random, sys

HOLES = {'DOOR_1': ('R1', 'R2'), 'DOOR_0': ('R1', 'R3'), 'DOOR_2': ('R3', 'R2')}
DIST = {'DOOR_1': 3.0, 'DOOR_0': 5.0, 'DOOR_2': 4.0}          # metres from the robot (pick_door tie-break)
T_FRAME = {'R1': 0.0, 'R2': 1.0, 'R3': 2.0}                   # the true transform room <- door frame; recall prior = +0.05
LEGS = [('new', 'R2'), ('room_1', 'R1'), ('new', 'R3'), ('new', 'R2'), ('room_3', 'R3'), ('room_1', 'R1')]
RC, DC, LT, CT = 'room_concept', 'door_concept', 'ltsm_agent', 'controller'
TYPE_OWNER = {'transition': None, 'room': RC, 'door': DC, 'passage': LT, 'recall': LT, 'proposal': LT, 'archive': LT, 'agent': LT}
EDGE_OWNER = {'current': RC, 'proto': RC, 'match': LT}          # v2 adds 'exit': owner by target type (see Store.edge_owner)
LEG_HOLE = {0: 'DOOR_1', 2: 'DOOR_0', 3: 'DOOR_2'}               # --close-doors: the tour closes every other door of a 'new' leg
CENTER = {'R1': (0.0, 0.0), 'R2': (10.0, 0.0), 'R3': (5.0, 8.66)}   # v2 map model: true room centres in one frame (m)
ROOM_R = 4.8                                                     # a room is a disc of this radius (centres are >= 10 m apart)
BEYOND = 0.65                                                    # the cross target lies 65% of the way centre->far centre
SHARED_ATTR = {('proposal', 'verdict'): RC}                   # SPEC GAP: spec says RC writes "rejected" ON the proposal


def holes_of(r): return [h for h, (a, b) in HOLES.items() if r in (a, b)]
def other_room(h, r): a, b = HOLES[h]; return b if r == a else a


class Violation(Exception):
    def __init__(s, inv, msg): super().__init__(msg); s.inv = inv; s.msg = msg


# ------------------------------------------------------------------------------------------------ graph
class View:
    def __init__(s, cascade=False):
        s.cascade = cascade                                      # v2: deleting a node drops the edges touching it (cortex)
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
            else:
                s.attrs.pop(key[1], None)
                if s.cascade:
                    for d in s.edges.values():
                        for pr in [q for q in d if key[1] in q]: d.pop(pr)
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
        s.rng = rng; s.cfg = cfg; s.seq = 0; s.nid = 0; s.auth = View(cfg.v2); s.views = {a: View(cfg.v2) for a in agents}
        s.log = []; s.last_write = 0; s.t = 0; s.T_reads = []; s.lag = cfg.lag_map
        s.types = {}; s.archived = set(); s.stats = collections.Counter(); s.restores = []

    def new_id(s): s.nid += 1; return s.nid

    def freeze_check(s, who, ops):
        a = s.auth; toks = sorted(a.by_type['transition'])
        creating = [val for key, val in ops if key[0] == 'n' and val and val['type'] == 'transition']
        if creating and toks: raise Violation(16, f"{who} opened a second transition token while n{toks[0]} ({a.attr(toks[0], 'state')}) exists")
        if len(toks) > 1: raise Violation(16, "more than one transition token")
        if creating:
            kind = next(v for k, v in ops if k[0] == 'a' and k[2] == 'kind'); st = next(v for k, v in ops if k[0] == 'a' and k[2] == 'state')
            if st != ORDER[kind][0]: raise Violation(18, f"token opened in state {st}, not {ORDER[kind][0]}")
        if toks and not creating:
            tok = toks[0]; st = a.attr(tok, 'state'); actor = ACTOR[st]
            for key, val in ops:
                if key[1] == tok and (key[0] in ('n', 'a')):
                    if key[0] == 'n' and val is None and (who != actor or st != ORDER[a.attr(tok, 'kind')][-1]):
                        raise Violation(18, f"token destroyed by {who} in state {st} (not the final step)")
                    continue
                if key[0] == 'e' or key[0] == 'n' or (key[0] == 'a' and key[2] == 'name'):
                    if who != actor:
                        raise Violation(17, f"STRUCTURAL write by {who} ({s._fmt(key, val)}) while the token says {st} (turn of {actor})")

    def lookup_by_name(s, who, name):
        """R5 instrumentation: a consumer resolved a room/door by NAME.  Fails once any rename has happened (names are display only)."""
        s.stats['lookup_name'] += 1
        if s.stats['renames']: raise Violation(20, f"{who} resolved '{name}' by NAME after a rename (identity must go through room_id)")

    def edge_owner(s, key):
        if key[1] == 'exit': return RC if s.types.get(key[3]) == 'room' else DC     # OLD_FACE->NEW_ROOM: room_concept; face->face: door_concept
        return EDGE_OWNER[key[1]]

    def check_owner(s, who, key, val):
        a = s.auth
        if key[0] == 'n':
            node = a.nodes.get(key[1])
            if node is None:
                if val is None: return
                if val['type'] == 'transition':
                    if val['owner'] != who or who not in (RC, LT): raise Violation(1, f"{who} created a token it may not open")
                    return
                if TYPE_OWNER[val['type']] != who or val['owner'] != who:
                    raise Violation(1, f"{who} created {val['type']} node {key[1]} owned by {TYPE_OWNER[val['type']]}")
            elif node['owner'] != who:
                raise Violation(1, f"{who} {'deleted' if val is None else 'rewrote'} {node['type']} node {key[1]} "
                                   f"({a.name(key[1])}) owned by {node['owner']}")
        elif key[0] == 'a':
            node = a.nodes.get(key[1])
            if node is None: return
            if node['type'] == 'transition' and key[2] == 'state':
                cur = node and a.attr(key[1], 'state'); order = ORDER[a.attr(key[1], 'kind')]
                if cur is None or val not in order: raise Violation(18, f"bad token state {val}")
                if who != ACTOR[cur]: raise Violation(18, f"{who} advanced the token in state {cur} (turn of {ACTOR[cur]})")
                if order.index(val) != order.index(cur) + 1:
                    raise Violation(18, f"token moved {cur} -> {val} ({'backwards' if order.index(val) <= order.index(cur) else 'skipping a step'})")
                return
            if node['owner'] != who and SHARED_ATTR.get((node['type'], key[2])) != who:
                raise Violation(1, f"{who} wrote attr {key[2]} of {node['type']} {key[1]} owned by {node['owner']}")
        else:
            if s.edge_owner(key) != who: raise Violation(1, f"{who} wrote {key[1]} edge owned by {s.edge_owner(key)}")

    def write(s, who, ops, t, note=''):
        for key, val in ops:
            if key[0] == 'n' and val: s.types[key[1]] = val['type']
        if s.cfg.v2t: s.freeze_check(who, ops)
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
                                                frame_T=m['frame_T'], mkey=ok, **(dict(rows=m.get('rows'), state='certain') if s.cfg.v2 else {}))), t)
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
                            T_prior=m['frame_T'] + 0.05, **(dict(rows=m.get('rows')) if s.cfg.v2 else {}))), t)
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
            out.append(dict(name=v.name(f), id=f, recalls=rec, match=m, entry=bool(v.attr(f, 'entry')), dist=DIST[v.attr(f, 'hole')], hole=v.attr(f, 'hole')))
        return out

    def pick(s, rel, doors):
        best = None
        for d in doors:
            if rel == 'new':
                if s.cfg.close_doors and d['hole'] != LEG_HOLE.get(s.leg): continue
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


# ================================================================================================ PROTOCOL v2 (Revision 2, 2026-10-30)
# R1 `exit` edges: room_concept writes OLD_FACE->NEW_ROOM at the switch; door_concept answers NEW_FACE->OLD_FACE.
# R2 the transition is OPEN from the first edge until a certain `recall_<OLD>` hangs under NEW_FACE; each owner then deletes its own left-room nodes.
# R3 ltsm predicts K behind a door from the map layout (seam errors, --map-err) and posts a predicted recall; room_concept verifies it.
# R4 door rows come only from ltsm memory (recall / accepted proposal), never from door_concept's own csv.
# Interpretations the mock had to make are marked "SPEC GAP" (see protocol_mock_v2_report.md).
def exits_valid(v):
    """(door a, room b) pairs of the OLD_FACE->NEW_ROOM kind whose endpoints exist in this view."""
    return [(a, b) for (a, b) in v.edges['exit']
            if a in v.nodes and b in v.nodes and v.nodes[a]['type'] == 'door' and v.nodes[b]['type'] == 'room']


def closing_ok(v, c, cfg):      # SPEC GAP: a PREDICTED recall_<OLD> under NEW_FACE must not close the phase (--closing-any = spec-literal)
    return cfg.closing_any or v.attr(c, 'state') != 'predicted'


def closed_for(v, old, b, cfg):
    if old not in v.nodes: return True
    rid = v.attr(old, 'room_id')                                     # R5: the closing recall names the room by ID (its `mkey` attr), never by name
    return any(v.attr(c, 'mkey') == rid and closing_ok(v, c, cfg) for f in v.kids_of(b, 'door') for c in v.kids_of(f, 'recall'))


def open_list(v, cfg):
    return [(a, b) for (a, b) in exits_valid(v) if not closed_for(v, v.nodes[a]['parent'], b, cfg)]


def is_open(v, cfg):
    """Is a structural transition in progress in this view?  v2: an exit edge without its closing recall; v2t: a token node exists."""
    return list(v.by_type['transition']) if cfg.v2t else open_list(v, cfg)


def left_closed(v, cur, cfg):
    """Rooms other than the current one whose closing fact (recall_<name> under a face of cur) is visible.
    SPEC GAP: found by NAME, not through the exit edge, because the edge vanishes when OLD_FACE is deleted by door_concept."""
    out = []
    for r in sorted(v.by_type['room']):
        if r == cur: continue
        rid = v.attr(r, 'room_id')
        if any(v.attr(c, 'mkey') == rid and closing_ok(v, c, cfg) for f in v.kids_of(cur, 'door') for c in v.kids_of(f, 'recall')):
            out.append(r)
    return out


def seam_err(cfg, ta, tb):
    """Error of the layout's transform room ta -> room tb: a fixed property of that seam (per seed), so schedule-independent."""
    a, b = sorted((ta, tb)); r = random.Random(f"{cfg.cur_seed}-{a}-{b}")
    e = (r.gauss(0, cfg.map_err), r.gauss(0, cfg.map_err))
    return e if (ta, tb) == (a, b) else (-e[0], -e[1])


class RoomConceptV2(RoomConcept):
    def init(s): RoomConcept.init(s); s.open_t = {}

    def cleanup(s, t):
        v = s.v; cur = v.current()
        if cur not in v.nodes: return
        for r in left_closed(v, cur, s.cfg):
            if r not in s.S.archived: raise Violation(14, f"room_concept deleted left room n{r} ({v.name(r)}) before ltsm archived it (closing fact misread)")
            ops = [(('e', 'exit', a, b), None) for (a, b) in exits_valid(v) if v.nodes[a]['parent'] == r]
            ops.append((('n', r), None))
            if v.proto(r): ops.append((('e', 'proto', r, r), None))
            s.put(ops, t); s.count_transition(r, t)

    def issued_id(s):
        ag = next(iter(s.v.by_type['agent']), None)
        return s.v.attr(ag, 'next_room_id') if ag is not None else None

    def count_transition(s, r, t):
        if r in s.open_t: s.S.stats['trans_n'] += 1; s.S.stats['trans_ticks'] += t - s.open_t.pop(r)

    def step(s, t):
        v, w, cfg = s.v, s.w, s.cfg
        if not s.started:
            rid = s.issued_id()
            if rid is None: return                                   # R5: wait for ltsm's next free room_id
            s.started = True
            s.new_room(t, f"room_{rid}", 'R1', dict(state='normal', outline='R1', frame_T=0.0, room_id=rid)); return
        s.cleanup(t)
        if w.epoch > s.handled and t >= w.inside_ready: s.switch(t)
        cur = v.current()
        if cur in v.nodes and v.proto(cur):
            e = v.attr(cur, 'est_steps', 0)
            if e < cfg.publishable:
                ops = [(('a', cur, 'est_steps'), e + 1)]
                if e + 1 >= 3 and v.attr(cur, 'outline') is None:
                    ops += [(('a', cur, 'outline'), v.nodes[cur]['truth']), (('a', cur, 'frame_T'), T_FRAME[v.nodes[cur]['truth']])]
                if e + 1 >= cfg.publishable: ops.append((('e', 'proto', cur, cur), None))     # first publishable: P stays a NEW room
                s.put(ops, t)
            for p in v.kids_of(cur, 'proposal'):
                if v.attr(p, 'verdict') is not None: continue
                if v.attr(p, 'target_outline') == v.attr(cur, 'outline'):
                    tt = T_FRAME[v.nodes[p]['truth']]
                    s.put([(('a', cur, 'name'), v.attr(p, 'target_name')), (('a', cur, 'outline'), v.attr(p, 'target_outline')),
                           (('a', cur, 'frame_T'), tt), (('a', cur, 'accepted_T'), tt), (('a', cur, 'room_id'), v.attr(p, 'target_mkey')),
                           (('a', cur, 'rows'), v.attr(p, 'rows')), (('a', cur, 'rows_src'), 'proposal'),
                           (('a', cur, 'state'), 'adopted'), (('e', 'proto', cur, cur), None), (('a', p, 'verdict'), 'accepted')], t)
                    s.renames.append((cur, v.attr(p, 'target_name'))); s.S.stats['renames'] += 1
                else: s.put([(('a', p, 'verdict'), 'rejected')], t)

    def verify(s, rec):
        """Model of the adoption test run as a test of the GIVEN pose (prior_only): the scans reject a wrong room outright,
        and accept the pose iff the true map error is inside the predicted covariance's 3-sigma."""
        v = s.v
        return v.nodes[rec]['truth'] == s.w.true_room and v.attr(rec, 'oracle_err') / v.attr(rec, 'sigma') <= 3.0

    def switch(s, t):
        v, w, cfg = s.v, s.w, s.cfg
        old = v.current(); hole = w.passed_hole
        if old not in v.nodes: return
        face = next((f for f in v.kids_of(old, 'door') if v.attr(f, 'hole') == hole), None)
        if face is None: return                         # wait for the FACT
        if s.issued_id() is None: return                # R5: wait for ltsm's next free room_id
        s.handled = w.epoch
        a = s.S.auth
        recs = v.kids_of(face, 'recall')
        certain = [r for r in recs if v.attr(r, 'state') != 'predicted']
        pred = [r for r in recs if v.attr(r, 'state') == 'predicted']
        rec = certain[0] if certain else (pred[0] if pred else None)
        case1 = rec is not None
        if rec is not None and rec in pred and not certain:
            if s.verify(rec): s.S.stats['pred_accept'] += 1
            else: s.S.stats['pred_reject'] += 1; case1 = False; s.note(t, f"predicted recall n{rec} REJECTED -> provisional")
        s.open_t[old] = t; s.before_room(t, face, old)
        if not case1:
            if rec is None:
                for f in a.kids_of(old, 'door'):
                    if a.attr(f, 'hole') == hole:
                        if [c for c in a.kids_of(f, 'recall') if a.attr(c, 'state') != 'predicted']:
                            raise Violation(2, f"room_concept took CASE 2 (provisional room) at {hole} but ltsm had ALREADY written a "
                                               f"certain recall under face n{f}; not yet visible to room_concept (acted on absence of a fact)")
                        if a.kids_of(f, 'recall'): s.S.stats['pred_late'] += 1
            rid = s.issued_id(); nm = f"room_{rid}"
            i = s.new_room(t, nm, w.true_room, dict(state='provisional', est_steps=0, room_id=rid), old=old, prov=True)
            s.created_prov[i] = w.true_room; s.note(t, f"CASE2 provisional {nm} n{i} truth={w.true_room}")
        else:
            tr = v.nodes[rec]['truth']
            if rec in certain and tr != w.true_room: raise Violation(5, f"recall n{rec} names {tr} but the robot is truly in {w.true_room}")
            nm = v.attr(rec, 'room_name')
            i = s.new_room(t, nm, tr, dict(state='recalled', outline=v.attr(rec, 'outline'), frame_T=v.attr(rec, 'frame_T'),
                                           room_id=v.attr(rec, 'mkey'), src_recall=rec), old=old)
            s.note(t, f"CASE1 recalled {nm} n{i} from {'predicted ' if rec in pred else ''}recall n{rec}")
        s.after_room(face, i, old, t)

    def before_room(s, t, face, old): pass

    def after_room(s, face, i, old, t):
        if s.cfg.fault != 'no_exit_edge': s.put([(('e', 'exit', face, i), True)], t)        # R1: the crossing, written ONCE, as its own batch


class DoorConceptV2(DoorConcept):
    def init(s): DoorConcept.init(s); s.csv = {}; s.entered = set(); s.merged = set()

    def make_face(s, v, cur, hole, name, t, aff, T=None):
        if aff and is_open(s.S.auth, s.cfg):
            raise Violation(11, f"door_concept OFFERED an affordance on a new face ({name}) while a transition is open (its view did not show it)")
        attrs = dict(name=name, hole=hole, aff=1 if aff else 0)
        if T is not None: attrs['T_used'] = T
        fid = s.S.new_id(); s.put(node_ops(fid, 'door', cur, DC, None, attrs), t); return fid

    def restore(s, v, cur, t, aff):
        """R4: remembered doors of a RECALLED/ADOPTED room come from ltsm's memory (recall / accepted proposal). False = waiting for that fact."""
        state = v.attr(cur, 'state'); rows = None; src = None
        if state == 'recalled':
            rc = v.attr(cur, 'src_recall')
            if rc is None or rc not in v.nodes: return False
            rows = dict(v.attr(rc, 'rows') or ()); src = 'recall'; T = v.attr(cur, 'frame_T')
        elif state == 'adopted' and v.attr(cur, 'accepted_T') is not None:
            if cur in s.merged: return True
            rows = dict(v.attr(cur, 'rows') or ()); src = v.attr(cur, 'rows_src') or 'proposal'; T = s.tfor(v, cur, state)
        if rows is None: return True
        if s.cfg.fault == 'door_rows_from_csv': rows = dict(s.csv.get(v.attr(cur, 'outline'), {})); src = 'csv'
        s.S.restores.append((s.name, cur, src))
        faces = {v.attr(f, 'hole'): f for f in v.kids_of(cur, 'door')}
        if state == 'adopted':
            for h, f in faces.items():
                if h in rows and v.name(f) != rows[h] and rows[h] not in {v.name(x) for x in v.by_type['door']}: s.put([(('a', f, 'name'), rows[h])], t)
                if v.attr(f, 'T_used') != T: s.put([(('a', f, 'T_used'), T)], t)
            s.merged.add(cur)
        for h, nm in rows.items():
            if h not in faces: faces[h] = s.make_face(v, cur, h, nm, t, aff, T)
        return True

    def step(s, t):
        if s.cfg.fault == 'no_exit_edge': return DoorConcept.step(s, t)
        v, w, cfg = s.v, s.w, s.cfg
        cur = v.current()
        if cur in v.nodes:
            for r in left_closed(v, cur, cfg):
                if r not in s.S.archived: raise Violation(14, f"door_concept deleted the doors of left room n{r} before ltsm archived it")
                for f in v.kids_of(r, 'door'): s.put([(('n', f), None)], t)
        for f in sorted(v.by_type['door']):
            p = v.nodes[f]['parent']
            if p not in v.nodes:
                if p not in s.S.archived: raise Violation(14, f"door_concept deleted orphan face n{f} whose room n{p} was never archived")
                s.put([(('n', f), None)], t)
        if cur not in v.nodes or v.attr(cur, 'entered_epoch') != w.epoch: return
        opens = is_open(v, cfg); hold = cfg.fault != 'no_transition_hold'
        aff = (not opens) if hold else True
        if w.epoch == 0: s.entered.add(cur)
        if cur not in s.entered:
            ex = [(a, b) for (a, b) in exits_valid(v) if b == cur]
            if not ex: return                                       # nothing happens in a new room until its exit edge is seen
            a, b = ex[0]
            if not s.restore(v, cur, t, False): return             # waiting for ltsm's memory fact
            hole = v.attr(a, 'hole'); faces = {v.attr(f, 'hole'): f for f in v.kids_of(cur, 'door')}
            nf = faces.get(hole) or s.make_face(v, cur, hole, s.fresh(v), t, False if hold else True)
            if (nf, a) not in v.edges['exit']: s.put([(('e', 'exit', nf, a), True)], t)         # R1: "this face is the far side of that hole"
            s.entered.add(cur)
        else: s.restore(v, cur, t, False)
        faces = {v.attr(f, 'hole'): f for f in v.kids_of(cur, 'door')}
        for h in w.sensed(t):
            if h not in faces: faces[h] = s.make_face(v, cur, h, s.fresh(v), t, aff)
        if cfg.fault == 'door_rows_from_csv': s.csv.setdefault(v.attr(cur, 'outline'), {}).update({h: v.name(f) for h, f in faces.items()})
        if aff:                                                     # offer (approach/cross/look) only when no transition is open
            for f in v.kids_of(cur, 'door'):
                if v.attr(f, 'aff') != 1:
                    if is_open(s.S.auth, cfg): raise Violation(11, f"door_concept OFFERED face {v.name(f)} while a transition is open (its view did not show it)")
                    s.put([(('a', f, 'aff'), 1)], t)


class LtsmV2(Ltsm):
    def init(s): Ltsm.init(s); s.done_tr = set(); s.key_truth = {}; s.settled = set(); s.agent = None; s.issued = None

    def assert_exit(s, p):
        a = s.S.auth; fa, fb, new, o = p.get('fa'), p.get('fb'), p['new'], p['old']
        ok = (fa is not None and fb is not None and (fa, new) in a.edges['exit'] and (fb, fa) in a.edges['exit']
              and a.nodes.get(fa, {}).get('parent') == o['id'] and a.nodes.get(fb, {}).get('parent') == new)
        if not ok: raise Violation(10, f"ltsm made a passage {o['name']} -> n{new} without an exit-edge pair (paired by inference)")

    def commit(s, p, t):
        v = s.v; o = p['old']; new = p['new']; hole = p['hole']; nk = s.key_for(new)
        s.assert_exit(p)
        s.rooms[o['key']] = dict(name=o['name'], outline=o['outline'], frame_T=o['frame_T'], truth=o['truth'], rows=o['rows'])
        s.key_truth[o['key']] = o['truth']; s.key_truth[nk] = v.nodes[new]['truth']
        pid = next((i for i, q in s.passages.items() if any(e[0] == o['key'] and e[1] == hole for e in q['ends'])), None)
        if pid is None:
            pid = len(s.passages); s.passages[pid] = dict(ends=[[o['key'], hole], [nk, hole]], crossings=1)
            nid = s.S.new_id(); s.pass_node[pid] = nid
            s.put(node_ops(nid, 'passage', None, LT, None, dict(name=f"passage_{pid}", ends=tuple(tuple(e) for e in s.passages[pid]['ends']),
                                                                   crossings=1)), t)
        else:
            s.passages[pid]['crossings'] += 1
            s.put([(('a', s.pass_node[pid], 'crossings'), s.passages[pid]['crossings'])], t)
        s.S.archived.add(o['id'])
        aid = s.S.new_id(); s.put(node_ops(aid, 'archive', None, LT, None, dict(name=f"archive_{o['id']}", node=o['id'])), t)
        if s.cfg.fault == 'ltsm_deletes_old_room': s.put([(('n', o['id']), None)], t)       # the old eviction
        s.settled.add(new); s.reconcile(new, t)                      # closing fact: recall_<OLD> under NEW_FACE, AFTER archive + passage
        s.note(t, f"archived {o['name']} key={o['key']} via {hole}; passage_{pid}")

    def finish_transition(s, t):                                     # only the no_exit_edge fault reaches the v1 trigger
        p = s.pending; new = p['new']; hole = p['hole']
        if new not in s.v.nodes: return
        if not any(s.v.attr(f, 'hole') == hole for f in s.v.kids_of(new, 'door')): return
        s.commit(p, t); s.pending = None

    def key_for(s, r):                                               # R5: the memory key IS the room_id
        k = s.v.attr(r, 'room_id'); s.node_key.setdefault(r, k); return k

    def carry_over(s, r, t):                                         # a provisional room adopted as K returns its number: passages follow
        v = s.v
        if r in s.carried or v.attr(r, 'state') != 'adopted' or v.attr(r, 'room_id') is None: return
        mp = s.node_key.get(r); x = v.attr(r, 'room_id'); s.carried.add(r); s.node_key[r] = x
        if mp is None or mp == x: return
        for pid, p in s.passages.items():
            for e in p['ends']:
                if e[0] == mp: e[0] = x
            s.put([(('a', s.pass_node[pid], 'ends'), tuple(tuple(e) for e in p['ends']))], t)
        s.key_truth[x] = s.key_truth.get(mp); s.note(t, f"carry-over {mp}->{x}")

    def issue(s, t):
        """R5: ltsm owns the identity counter: the smallest room_id neither in memory nor on a live room; published on its own agent node."""
        v = s.v
        used = set(s.rooms) | {v.attr(r, 'room_id') for r in v.by_type['room']}
        nxt = next(i for i in range(1, 99) if i not in used)
        if s.agent is None:
            s.agent = s.S.new_id(); s.put(node_ops(s.agent, 'agent', None, LT, None, dict(name='ltsm_agent', next_room_id=nxt)), t); s.issued = nxt
        elif nxt != s.issued: s.put([(('a', s.agent, 'next_room_id'), nxt)], t); s.issued = nxt

    def snap(s, r):
        d = Ltsm.snap(s, r); v = s.v
        d['rows'] = tuple(sorted((v.attr(f, 'hole'), v.name(f)) for f in v.kids_of(r, 'door'))); return d

    def predict(s, cur, kc, f):
        v = s.v; tc = v.nodes[cur]['truth']; h = v.attr(f, 'hole'); tf = other_room(h, tc)
        cc, cf = CENTER[tc], CENTER[tf]
        P = ((cf[0] - cc[0]) * BEYOND, (cf[1] - cc[1]) * BEYOND)     # the point just beyond the door, in cur's frame (exact: from the live door)
        est = {kc: ((0.0, 0.0), 0)}; q = [kc]
        while q:                                                     # layout: rooms placed through the passages' seams (BFS = fewest seams)
            x = q.pop(0); (px, py), d = est[x]; tx = s.key_truth.get(x)
            for p in s.passages.values():
                (ka, _), (kb, _) = p['ends']
                if ka == kb: continue
                for u, y in ((ka, kb), (kb, ka)):
                    ty = s.key_truth.get(y)
                    if u != x or y in est or tx is None or ty is None: continue
                    e = seam_err(s.cfg, tx, ty)
                    est[y] = ((px + CENTER[ty][0] - CENTER[tx][0] + e[0], py + CENTER[ty][1] - CENTER[tx][1] + e[1]), d + 1); q.append(y)
        best = None
        for x, ((ex, ey), d) in est.items():
            if x == kc or x not in s.rooms: continue
            dist = math.hypot(P[0] - ex, P[1] - ey)
            if dist < ROOM_R and (best is None or dist < best[0]): best = (dist, x, d, ex, ey)
        if best is None: return None
        _, x, d, ex, ey = best; tx = s.rooms[x]['truth']
        err = math.hypot(ex - (CENTER[tx][0] - cc[0]), ey - (CENTER[tx][1] - cc[1]))
        return x, d, err

    def step(s, t):
        v, cfg = s.v, s.cfg
        s.issue(t)
        if cfg.fault == 'no_exit_edge':
            s.settled.add(v.current()); return Ltsm.step(s, t)
        for kind in ('recall', 'proposal'):
            for n in sorted(v.by_type[kind]):
                if v.nodes[n]['parent'] not in v.nodes: s.put([(('n', n), None)], t)
        for (pn, f), _ in list(v.edges['match'].items()):
            if f not in v.nodes: s.put([(('e', 'match', pn, f), None)], t)
        cur = v.current()
        if cur in v.nodes and not s.settled and s.cur is None: s.cur = cur; s.key_for(cur); s.settled.add(cur)
        for (a, b) in sorted(exits_valid(v)):                        # R1: pair the two faces from the exit edges ONLY
            if a in s.done_tr: continue
            old = v.nodes[a]['parent']
            fb = next((f for f in v.kids_of(b, 'door') if (f, a) in v.edges['exit']), None)
            if old not in v.nodes or fb is None: continue            # wait for the far face's answering edge
            s.carry_over(old, t)
            sn = s.snap(old); sn['key'] = s.key_for(old)
            s.done_tr.add(a); s.commit(dict(old=sn, new=b, hole=v.attr(a, 'hole'), fa=a, fb=fb), t)
        cur = v.current()
        if cur in v.nodes:
            s.key_truth[s.key_for(cur)] = v.nodes[cur]['truth']
            s.carry_over(cur, t)
            if cur in s.settled:
                s.reconcile(cur, t)
                if not cfg.no_map and not is_open(v, cfg): s.predict_doors(cur, t)
                s.recognise(cur, t)

    def predict_doors(s, cur, t):
        v = s.v; kc = s.key_for(cur)
        crossed = {a for (a, b) in exits_valid(v)} | {a for (a, b) in v.edges['exit'] if a in v.nodes and b in v.nodes and v.nodes[b]['type'] == 'door'}
        for f in v.kids_of(cur, 'door'):
            h = v.attr(f, 'hole')
            if v.kids_of(f, 'recall') or f in crossed or any(e[0] == kc and e[1] == h for q in s.passages.values() for e in q['ends']): continue
            pr = s.predict(cur, kc, f)
            if pr is None: continue
            x, d, err = pr; m = s.rooms[x]
            s.put(node_ops(s.S.new_id(), 'recall', f, LT, m['truth'],
                           dict(name=f"recall_{m['name']}", room_name=m['name'], outline=m['outline'], frame_T=m['frame_T'] + err, mkey=x,
                                rows=m.get('rows'), state='predicted', sigma=s.cfg.map_sigma * math.sqrt(d), oracle_err=err)), t)
            s.note(t, f"PREDICTED recall_{m['name']} under n{f} ({d} seams, err={err:.2f} m, sigma={s.cfg.map_sigma * math.sqrt(d):.2f})")


class ControllerV2(Controller):
    def init(s):
        Controller.init(s); s.chosen = None
        # R5: a mission that names room_N is resolved to its room_id ONCE, when accepted; display names are room_<room_id>
        s.mission = {rel: int(rel[5:]) for rel, _ in LEGS if rel.startswith('room_')}; s.S.stats['lookup_name_at_accept'] += len(s.mission)

    def facts(s, cur):
        if s.cfg.fault == 'no_exit_edge': return Controller.facts(s, cur)
        v = s.v; out = []
        for f in v.kids_of(cur, 'door'):
            recs = [(v.attr(c, 'room_name'), v.attr(c, 'state')) for c in v.kids_of(f, 'recall')]
            out.append(dict(name=v.name(f), id=f, hole=v.attr(f, 'hole'), aff=bool(v.attr(f, 'aff')), rooms={n for n, _ in recs},
                            room_ids={v.attr(c, 'mkey') for c in v.kids_of(f, 'recall')},
                            certain=any(st != 'predicted' for _, st in recs), dist=DIST[v.attr(f, 'hole')],
                            exit_out=any(x == f and y in v.nodes and v.nodes[y]['type'] == 'door' for (x, y) in v.edges['exit'])))
        return out

    def pick(s, rel, doors):
        if s.cfg.fault == 'no_exit_edge': return Controller.pick(s, rel, doors)
        best = None
        for d in doors:
            if s.cfg.fault != 'no_transition_hold' and not d['aff']: continue
            if rel == 'new':
                if s.cfg.close_doors and d['hole'] != LEG_HOLE.get(s.leg): continue
                if d['exit_out'] or (d['certain'] and not s.cfg.spec_literal): continue    # SPEC GAP: E2 dies with OLD_FACE; the certain recall remains
                if best is None or (d['dist'], d['name']) < (best['dist'], best['name']): best = d
            else:
                if s.cfg.fault == 'name_keyed_lookup': s.S.lookup_by_name(s.name, rel); hit = rel in d['rooms']
                else: s.S.stats['lookup_id'] += 1; hit = s.mission[rel] in d['room_ids']
                if hit and (best is None or d['name'] < best['name']): best = d
        return best

    def step(s, t):
        if s.cfg.fault == 'no_exit_edge': return Controller.step(s, t)
        if s.leg >= len(LEGS): s.done = True; return
        v, w, cfg = s.v, s.w, s.cfg; rel, expect = LEGS[s.leg]; cur = v.current()
        if s.phase == 'start':
            if cur in v.nodes: s.phase = 'explore'; s.timer = cfg.explore; s.start_id = cur
        elif s.phase == 'explore':
            s.timer -= 1
            if s.timer <= 0: s.phase = 'hold'; s.since = t
        elif s.phase == 'hold':
            if cur in v.nodes and cfg.fault != 'no_transition_hold' and is_open(v, cfg):
                s.wait_desc = f"leg {s.leg + 1} cross {rel}: transition open (waiting for its closing recall)"
                if t - s.since > cfg.hold: s.stuck(t)
                return
            d = s.pick(rel, s.facts(cur)) if cur in v.nodes else None
            if d is None:
                s.wait_desc = f"leg {s.leg + 1} cross {rel}: no door resolves in {v.name(cur)}"
                if t - s.since > cfg.hold: s.stuck(t)
                return
            a = s.S.auth
            if cur not in a.nodes or a.nodes[cur]['truth'] != w.true_room:
                raise Violation(5, f"controller (room {v.name(cur)}) drives while the robot is truly in {w.true_room}")
            if d['id'] not in a.nodes: raise Violation(2, f"controller drives at door n{d['id']} that no longer exists")
            if is_open(a, cfg): raise Violation(11, f"controller SELECTED door {d['name']} while a transition is open (its view did not show it)")
            if rel == 'new':
                af = d['id']
                if any(x == af and y in a.nodes and a.nodes[y]['type'] == 'door' for (x, y) in a.edges['exit']) or \
                        [c for c in a.kids_of(af, 'recall') if a.attr(c, 'state') != 'predicted']:
                    raise Violation(2, f"controller took {d['name']} as 'new' but it is the way back (exit edge / certain recall already in the graph)")
            hole = a.attr(d['id'], 'hole')
            if hole not in holes_of(w.true_room): raise Violation(5, f"door {d['name']} ({hole}) is not in the true room {w.true_room}")
            s.chosen = d['id']; w.start_cross(hole, t); s.phase = 'drive'
        elif s.phase == 'drive':
            if w.crossing is None: s.phase = 'switch'; s.since = t
        elif s.phase == 'switch':
            ex = [b for (x, b) in exits_valid(v) if x == s.chosen]
            if not ex and s.chosen not in v.nodes and cur in v.nodes and cur != s.start_id:
                ex = [cur]            # SPEC GAP: the exit edge is deleted within the transition (cleanup); if the controller saw it late, the durable fact is the room switch
                s.S.stats['ctl_missed_exit'] += 1
            if ex:                                                   # the leg completes on the exit edge (fact 3), nothing else
                b = ex[0]; a = s.S.auth
                if a.nodes[b]['truth'] != w.true_room:
                    raise Violation(5, f"leg {s.leg + 1} done: exit edge leads to {v.name(b)} ({a.nodes[b]['truth']}), robot truly in {w.true_room}")
                if w.true_room != expect: raise Violation(5, f"leg {s.leg + 1} (cross {rel}) landed in {w.true_room}, expected {expect}")
                s.legs_done.append((s.leg + 1, t, v.name(b), b)); s.note(t, f"LEG {s.leg + 1} done in {v.name(b)} n{b}")
                s.leg += 1; s.phase = 'start'
            else:
                s.wait_desc = f"leg {s.leg + 1} waiting for the exit edge (fact 3)"
                if t - s.since > cfg.hold: s.stuck(t)


# ================================================================================================ PROTOCOL v2t (R2 as an explicit TRANSITION TOKEN)
# One `transition` node (metaconcept, `state` attribute).  While it exists only the agent its state names makes structural writes
# (node/edge create/delete/rename); everybody may still update attributes of their own nodes; no door is offered/selected.
ORDER = {'cross': ['cross:entering', 'cross:entered', 'cross:faced', 'cross:archived', 'cross:released'],
         'adopt': ['adopt:proposed', 'adopt:judged', 'adopt:seated']}
ACTOR = {'cross:entering': RC, 'cross:entered': DC, 'cross:faced': LT, 'cross:archived': DC, 'cross:released': RC,
         'adopt:proposed': RC, 'adopt:judged': DC, 'adopt:seated': LT}


def tokens(v): return sorted(v.by_type['transition'])


def advance(agent, tok, t):
    st = agent.v.attr(tok, 'state'); order = ORDER[agent.v.attr(tok, 'kind')]
    agent.put([(('a', tok, 'state'), order[order.index(st) + 1])], t)


class RoomConceptT(RoomConceptV2):
    def init(s): RoomConceptV2.init(s); s.tok_old = {}; s.tok = None

    def before_room(s, t, face, old):
        s.tok = s.S.new_id(); s.tok_old[s.tok] = old
        s.put(node_ops(s.tok, 'transition', None, RC, None, dict(name='transition', kind='cross', state='cross:entering')), t)

    def after_room(s, face, i, old, t):                              # still inside cross:entering: the exit edge, then hand over
        if s.cfg.fault != 'no_exit_edge': s.put([(('e', 'exit', face, i), True)], t)
        advance(s, s.tok, t)

    def step(s, t):
        v, w, cfg = s.v, s.w, s.cfg
        if not s.started:
            rid = s.issued_id()
            if rid is None: return
            s.started = True
            s.new_room(t, f"room_{rid}", 'R1', dict(state='normal', outline='R1', frame_T=0.0, room_id=rid)); return
        toks = tokens(v); cur = v.current()
        if toks:
            tok = toks[0]; st = v.attr(tok, 'state')
            if cfg.fault == 'no_freeze' and st == 'cross:faced' and tok in s.tok_old and s.tok_old[tok] in v.nodes:   # out of turn
                s.put([(('n', s.tok_old[tok]), None)], t); s.tok_old.pop(tok)
            elif st == 'cross:released' and tok in s.tok_old:
                r = s.tok_old.pop(tok)
                ops = [(('e', 'exit', a, b), None) for (a, b) in exits_valid(v) if v.nodes[a]['parent'] == r]
                if r in v.nodes:
                    ops.append((('n', r), None))
                    if v.proto(r): ops.append((('e', 'proto', r, r), None))
                ops.append((('n', tok), None)); s.put(ops, t); s.count_transition(r, t)
            elif st == 'adopt:proposed' and cur in v.nodes:
                ps = [p for p in v.kids_of(cur, 'proposal') if v.attr(p, 'verdict') is None]
                if ps:
                    p = ps[0]
                    if v.attr(p, 'target_outline') == v.attr(cur, 'outline'):
                        tt = T_FRAME[v.nodes[p]['truth']]
                        s.put([(('a', cur, 'name'), v.attr(p, 'target_name')), (('a', cur, 'outline'), v.attr(p, 'target_outline')),
                               (('a', cur, 'frame_T'), tt), (('a', cur, 'accepted_T'), tt), (('a', cur, 'room_id'), v.attr(p, 'target_mkey')),
                               (('a', cur, 'rows'), v.attr(p, 'rows')), (('a', cur, 'rows_src'), 'proposal'),
                               (('a', cur, 'state'), 'adopted'), (('e', 'proto', cur, cur), None), (('a', p, 'verdict'), 'accepted')], t)
                        s.renames.append((cur, v.attr(p, 'target_name'))); s.S.stats['renames'] += 1
                    else: s.put([(('a', p, 'verdict'), 'rejected')], t)
                    advance(s, tok, t)
        elif w.epoch > s.handled and t >= w.inside_ready: s.switch(t)
        cur = v.current()
        if cur in v.nodes and v.proto(cur):                          # perception continues: attribute updates are always allowed
            e = v.attr(cur, 'est_steps', 0)
            if e < cfg.publishable:
                ops = [(('a', cur, 'est_steps'), e + 1)]
                if e + 1 >= 3 and v.attr(cur, 'outline') is None:
                    ops += [(('a', cur, 'outline'), v.nodes[cur]['truth']), (('a', cur, 'frame_T'), T_FRAME[v.nodes[cur]['truth']])]
                if e + 1 >= cfg.publishable and not toks: ops.append((('e', 'proto', cur, cur), None))   # structural: deferred while a token exists
                s.put(ops, t)
            elif not toks: s.put([(('e', 'proto', cur, cur), None)], t)


class DoorConceptT(DoorConceptV2):
    def step(s, t):
        if s.cfg.fault == 'no_exit_edge': return DoorConcept.step(s, t)
        v, w, cfg = s.v, s.w, s.cfg
        cur = v.current(); toks = tokens(v); hold = cfg.fault != 'no_transition_hold'
        if toks and cur in v.nodes:
            tok = toks[0]; st = v.attr(tok, 'state')
            if st == 'cross:entered':
                ex = [(a, b) for (a, b) in exits_valid(v) if b == cur]
                if not ex: return
                a, b = ex[0]
                if not s.restore(v, cur, t, False): return
                hole = v.attr(a, 'hole'); faces = {v.attr(f, 'hole'): f for f in v.kids_of(cur, 'door')}
                nf = faces.get(hole) or s.make_face(v, cur, hole, s.fresh(v), t, False if hold else True)
                if (nf, a) not in v.edges['exit']: s.put([(('e', 'exit', nf, a), True)], t)
                s.entered.add(cur); advance(s, tok, t)
            elif st == 'cross:archived':
                olds = {v.nodes[a]['parent'] for (a, b) in exits_valid(v) if b == cur} or {r for r in v.by_type['room'] if r != cur}
                for r in sorted(olds):
                    for f in v.kids_of(r, 'door'): s.put([(('n', f), None)], t)
                advance(s, tok, t)
            elif st == 'adopt:judged':
                pr = v.kids_of(cur, 'proposal')
                if v.attr(cur, 'state') == 'adopted' and v.attr(cur, 'accepted_T') is not None:
                    if s.restore(v, cur, t, False): advance(s, tok, t)
                elif pr and all(v.attr(p, 'verdict') == 'rejected' for p in pr): advance(s, tok, t)
            return
        if toks: return
        for f in sorted(v.by_type['door']):
            p = v.nodes[f]['parent']
            if p not in v.nodes: s.put([(('n', f), None)], t)
        if cur not in v.nodes or v.attr(cur, 'entered_epoch') != w.epoch: return
        if w.epoch == 0: s.entered.add(cur)
        if cur not in s.entered: return
        faces = {v.attr(f, 'hole'): f for f in v.kids_of(cur, 'door')}
        for h in w.sensed(t):
            if h not in faces: faces[h] = s.make_face(v, cur, h, s.fresh(v), t, True)
        for f in v.kids_of(cur, 'door'):                             # no token: offer
            if v.attr(f, 'aff') != 1:
                if is_open(s.S.auth, cfg): raise Violation(11, f"door_concept OFFERED face {v.name(f)} while a transition token exists (its view did not show it)")
                s.put([(('a', f, 'aff'), 1)], t)


class LtsmT(LtsmV2):
    def step(s, t):
        if s.cfg.fault == 'no_exit_edge': return LtsmV2.step(s, t)
        return s.step_t(t)

    def recognise(s, cur, t):
        n = len(s.v.kids_of(cur, 'proposal')); Ltsm.recognise(s, cur, t)
        if len(s.v.kids_of(cur, 'proposal')) > n:                    # opened by ltsm when it posts a proposal
            s.tok = s.S.new_id()
            s.put(node_ops(s.tok, 'transition', None, LT, None, dict(name='transition', kind='adopt', state='adopt:proposed')), t)

    def step_t(s, t):
        v, cfg = s.v, s.cfg
        s.issue(t)
        toks = tokens(v); cur = v.current()
        if cur in v.nodes and not s.settled and s.cur is None: s.cur = cur; s.key_for(cur); s.settled.add(cur)
        if toks:
            tok = toks[0]; st = v.attr(tok, 'state')
            if st == 'cross:faced':
                for (a, b) in sorted(exits_valid(v)):
                    if a in s.done_tr: continue
                    old = v.nodes[a]['parent']
                    fb = next((f for f in v.kids_of(b, 'door') if (f, a) in v.edges['exit']), None)
                    if old not in v.nodes or fb is None: continue
                    s.carry_over(old, t); sn = s.snap(old); sn['key'] = s.key_for(old)
                    s.done_tr.add(a); s.commit(dict(old=sn, new=b, hole=v.attr(a, 'hole'), fa=a, fb=fb), t); advance(s, tok, t); break
            elif st == 'adopt:seated' and cur in v.nodes:
                s.carry_over(cur, t); s.put([(('n', tok), None)], t)
            if cur in v.nodes:
                s.key_truth[s.key_for(cur)] = v.nodes[cur]['truth']; s.carry_over(cur, t)
                if v.proto(cur) and v.attr(cur, 'outline') is not None and not v.kids_of(cur, 'proposal'): s.ev[cur] = s.ev.get(cur, 0) + 1   # perception continues
            return
        for kind in ('recall', 'proposal'):
            for n in sorted(v.by_type[kind]):
                if v.nodes[n]['parent'] not in v.nodes: s.put([(('n', n), None)], t)
        for (pn, f), _ in list(v.edges['match'].items()):
            if f not in v.nodes: s.put([(('e', 'match', pn, f), None)], t)
        if cur in v.nodes:
            s.key_truth[s.key_for(cur)] = v.nodes[cur]['truth']; s.carry_over(cur, t)
            if cur in s.settled:
                s.reconcile(cur, t)
                if not cfg.no_map: s.predict_doors(cur, t)
                s.recognise(cur, t)


# ------------------------------------------------------------------------------------------------ checks
def check_tick(S, w, agents, prov_ids):
    a = S.auth; t = S.t
    for who, src, attr, tt in S.T_reads:
        if attr not in ('accepted_T', 'frame_T'):
            raise Violation(2, f"{who} used transform '{attr}' of n{src} (a PROPOSAL prior) instead of the accepted/remembered one")
    S.T_reads.clear()
    for who, cur, src in S.restores:
        if src not in ('recall', 'proposal'): raise Violation(13, f"{who} restored the doors of room n{cur} from '{src}', not from ltsm's memory (recall/proposal)")
    S.restores.clear()
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
        if p not in a.nodes:
            if S.cfg.v2 and p in S.archived: continue               # v2: owners clean up independently on the closing fact; transient orphan
            raise Violation(4, f"door face n{f} ({a.name(f)}) has no parent room (n{p} gone)")
        h = a.attr(f, 'hole')
        if h not in holes_of(a.nodes[p]['truth']): raise Violation(4, f"face n{f} {a.name(f)} hole {h} is not a hole of its room {a.name(p)} ({a.nodes[p]['truth']})")
        if (p, h) in seen: raise Violation(4, f"two faces for hole {h} in room {a.name(p)}")
        seen.add((p, h))
    if S.cfg.v2:      # SPEC GAP: `recall_<room>` is not unique while the room left is still alive (DSR names are unique) -- counted, not failed
        rn = collections.Counter(a.name(n) for n in a.by_type['recall'])
        if any(c > 1 for c in rn.values()): S.stats['recall_name_dup'] = 1
    if S.cfg.v2:      # R5: one room_id per room, ever (live or in memory)
        ids = {}
        for r in a.by_type['room']:
            rid = a.attr(r, 'room_id')
            if rid in ids: raise Violation(19, f"two live rooms share room_id {rid}: {a.name(ids[rid])} and {a.name(r)}")
            ids[rid] = r
        for k, m in agents[LT].rooms.items():
            if k in ids and a.nodes[ids[k]]['truth'] != m['truth']:
                raise Violation(19, f"room_id {k} names two different rooms: live {a.name(ids[k])} ({a.nodes[ids[k]]['truth']}) vs memory {m['name']} ({m['truth']})")
    for typ in (('room', 'door', 'passage') if S.cfg.v2 else ('room', 'door')):
        names = collections.Counter(a.name(n) for n in a.by_type[typ])
        dup = [n for n, c in names.items() if c > 1]
        if dup: raise Violation(8, f"duplicate live {typ} names {dup}")
    for i in list(prov_ids):
        if i not in a.nodes:
            if (i in S.archived) if S.cfg.v2 else any(a.attr(x, 'room_id') == i for x in a.by_type['archive']): prov_ids.discard(i)
            else: raise Violation(6, f"provisional room n{i} destroyed (never renamed in place / left without an archive)")


def final_checks_v2(S, agents):
    lt = agents[LT]; a = S.auth; rc = agents[RC]
    outs = collections.Counter(m['outline'] for m in lt.rooms.values())
    if any(c > 1 for c in outs.values()): raise Violation(6, f"ltsm memory stores a room twice (never recognised): {dict(outs)}")
    if not (any(nm == 'room_2' for _, nm in rc.renames) or S.stats['pred_accept']):
        raise Violation(6, "the revisited room_2 was neither adopted in place nor entered through an accepted prediction")
    for pid, q in lt.passages.items():
        if q['ends'][0][0] == q['ends'][1][0]: raise Violation(9, f"memory passage_{pid} loops on one room")
    if len(lt.passages) != 3: raise Violation(9, f"memory holds {len(lt.passages)} passages, expected 3: {[q['ends'] for q in lt.passages.values()]}")
    if agents[DC].csv or rc.hwm:
        raise Violation(21, f"a memory wipe would leave identities in other agents: door_concept csv rows {len(agents[DC].csv)}, room_concept hwm {rc.hwm}")
    if a.by_type['transition']: raise Violation(15, f"transition token leaked: {[a.attr(x, 'state') for x in a.by_type['transition']]}")
    if len(a.by_type['room']) != 1 or any(a.nodes[f]['parent'] not in a.nodes for f in a.by_type['door']):
        raise Violation(15, f"left room / its doors leaked: rooms {[a.name(r) for r in a.by_type['room']]}, "
                            f"orphan faces {[a.name(f) for f in a.by_type['door'] if a.nodes[f]['parent'] not in a.nodes]}")
    if S.cfg.scenario == 'tour' and not S.stats['pred_accept']:
        raise Violation(12, f"leg 4 (known room_2 behind DOOR_2) was not entered through an accepted PREDICTION "
                            f"(pred_late={S.stats['pred_late']} pred_reject={S.stats['pred_reject']})")


def final_checks(S, agents):
    if S.cfg.v2: return final_checks_v2(S, agents)
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
    cfg = argparse.Namespace(**vars(cfg)); cfg.run_dmax = rng.randint(0, cfg.dmax); cfg.cur_seed = seed
    names = [RC, DC, LT, CT]
    S = Store(rng, cfg, names); w = World(rng, cfg)
    K = ((RoomConceptT, DoorConceptT, LtsmT, ControllerV2) if cfg.v2t else (RoomConceptV2, DoorConceptV2, LtsmV2, ControllerV2)) if cfg.v2 \
        else (RoomConcept, DoorConcept, Ltsm, Controller)
    ag = {RC: K[0](RC, S, w, rng, cfg), DC: K[1](DC, S, w, rng, cfg), LT: K[2](LT, S, w, rng, cfg), CT: K[3](CT, S, w, rng, cfg)}
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
        run_one.stats = S.stats
        return None, S.log
    except Violation as e:
        run_one.stats = S.stats
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
    ap.add_argument('--fault', choices=['name_keyed_lookup', 'pair_same_room', 'read_prior_not_accepted', 'ltsm_deletes_provisional',
                                        'no_exit_edge', 'no_transition_hold', 'door_rows_from_csv', 'ltsm_deletes_old_room', 'no_freeze'])
    ap.add_argument('--protocol', choices=['v1', 'v2', 'v2t'], default='v2')
    ap.add_argument('--scenario', choices=['tour', 'no_map', 'bad_map'], default='tour')
    ap.add_argument('--map-err', dest='map_err', type=float, default=None, help='std (m, per axis) of the TRUE error of each layout seam (default 0.2; bad_map 2.5)')
    ap.add_argument('--map-sigma', dest='map_sigma', type=float, default=0.3, help='ltsm\'s CLAIMED per-seam sigma (m); predicted sigma = this * sqrt(#seams)')
    ap.add_argument('--publishable', type=int, default=12, help='v2: room_concept steps until a provisional room is publishable (proto edge removed)')
    ap.add_argument('--close-doors', dest='close_doors', action='store_true', help='the tour closes every door a "new" leg must not take (removes the leg-1 harness artifact)')
    ap.add_argument('--closing-any', dest='closing_any', action='store_true', help='v2 spec-literal: ANY recall_<OLD> under NEW_FACE closes (even a predicted one)')
    ap.add_argument('--spec-literal', dest='spec_literal', action='store_true', help='v2 spec-literal: "cross new" excludes ONLY doors with an exit edge')
    ap.add_argument('--entry-mark', dest='entry_mark', action='store_true', help='mitigation: door_concept marks the entry face; "cross new" skips it')
    ap.add_argument('--stress', action='store_true', help='dmax 10, drop 12, stall 12')
    ap.add_argument('--tail', type=int, default=90, help='trace lines printed for the first failure')
    cfg = ap.parse_args()
    if cfg.stress: cfg.dmax, cfg.drop, cfg.stall = 10, 12, 12
    cfg.v2 = cfg.protocol in ('v2', 'v2t'); cfg.v2t = cfg.protocol == 'v2t'; cfg.no_map = cfg.scenario == 'no_map'
    if cfg.map_err is None: cfg.map_err = 2.5 if cfg.scenario == 'bad_map' else 0.2
    cfg.lag_map = {{'room': RC, 'door': DC, 'ltsm': LT, 'controller': CT}.get(k, k): int(v) for k, v in (x.split('=') for x in cfg.lag.split(',') if x)}
    if cfg.seed is not None:
        e, log = run_one(cfg.seed, cfg); print('\n'.join(log)); print('RESULT:', 'PASS' if e is None else f'FAIL inv {e.inv}'); return 0 if e is None else 1
    fails = collections.Counter(); first = None; ok = 0; agg = collections.Counter()
    for sd in range(cfg.seed0, cfg.seed0 + cfg.seeds):
        e, log = run_one(sd, cfg); agg.update(run_one.stats)
        if e is None: ok += 1
        else:
            fails[e.inv] += 1
            if first is None: first = (sd, e, log)
    print(f"protocol={cfg.protocol} scenario={cfg.scenario} fault={cfg.fault} entry_mark={cfg.entry_mark} dmax={cfg.dmax} stall={cfg.stall} explore={cfg.explore}: "
          f"{ok}/{cfg.seeds} pass, {cfg.seeds - ok} fail; failing invariants {dict(sorted(fails.items()))}")
    tp = agg['trans_ticks'] / agg['trans_n'] if agg['trans_n'] else float('nan')
    print(f"   ticks/transition {tp:.1f} (n={agg['trans_n']}); pred_accept {agg['pred_accept']} pred_reject {agg['pred_reject']} pred_late {agg['pred_late']}; "
          f"renames {agg['renames']}; runs with duplicate recall names {agg['recall_name_dup']}; controller missed exit edge {agg['ctl_missed_exit']}; "
          f"lookups by id {agg['lookup_id']} by name {agg['lookup_name']} (+{agg['lookup_name_at_accept']} at mission accept)")
    if first:
        sd, e, log = first
        print(f"-- first failure: seed {sd}, invariant {e.inv}: {e.msg}\n-- trace (last {cfg.tail} lines)")
        print('\n'.join(log[-cfg.tail:]))
    return 0 if (ok == cfg.seeds) == (cfg.fault is None) else 1


if __name__ == '__main__':
    sys.exit(main())
