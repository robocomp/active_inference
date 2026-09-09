#!/usr/bin/env python3
"""rt_twist_ring_check — is the twist actually RINGED, and does slot i of it belong to slot i
of the pose?

WHY THIS EXISTS. rt_twist_linear / rt_twist_angular (cortex, 2026-09-09) replace the loose
rt_translation_velocity pair. The whole claim of the change is that a twist now lands in the SAME
ring slot as the pose it was measured with, and therefore inherits that block's timestamp. That claim
is not visible from the agent's own logs and it fails SILENTLY: a twist packed against the wrong slot
is a well-formed 3-vector attached to the wrong instant. So it gets its own check.

It also verifies the ★AXIS ORDER swap, which is the other half of the change and the one that has
already cost the same 90-degree bug in three consumers: the ring pair is [x, y, z] in the child's own
axes while the legacy pair is ARRAY order [adv, side, _]. On a +Y-forward robot the first two numbers
must therefore appear SWAPPED between the two attributes. If they match instead, the producer wrote
array order into an axis-order attribute and every future consumer inherits the bug this replaced.

  usage:  python3 rt_twist_ring_check.py [--agent-id 900] [--seconds 6]

★pydsr needs a Qt event loop or the graph reads back EMPTY (0 nodes) — see the QTimer below. A run
that prints "no nodes" is almost always this, not an absent graph.
"""
import argparse, sys
from PySide6.QtCore import QCoreApplication, QTimer
import pydsr

ap = argparse.ArgumentParser()
ap.add_argument("--agent-id", type=int, default=907, help="must be UNIQUE across the live graph")
ap.add_argument("--agent-name", default="rt_twist_ring_check")
ap.add_argument("--robot", default="Shadow", help="the localisation frame (type 'robot')")
ap.add_argument("--seconds", type=int, default=6, help="how long to watch the ring advance")
a = ap.parse_args()

app = QCoreApplication(sys.argv)
G = pydsr.DSRGraph(0, a.agent_name, a.agent_id)

BLOCK, HIST = 3, 5

def find_rt_edge():
    """The edge is written parent->child and which is which depends on the anchoring, so try both."""
    robot = G.get_node(a.robot)
    if robot is None:
        return None, "no node named '%s'" % a.robot
    rooms = G.get_nodes_by_type("room")
    if not rooms:
        return None, "no node of type 'room'"
    for room in rooms:
        for src, dst in ((room.id, robot.id), (robot.id, room.id)):
            e = G.get_edge(src, dst, "RT")
            if e is not None:
                return e, "%d -> %d" % (src, dst)
    return None, "no RT edge between room and %s" % a.robot

def attr(e, name):
    # ★pydsr's attrs is a MapStringAttribute, NOT a dict: it has no .get, and indexing a missing key
    # raises. Membership-test first or every read of an absent attribute becomes a traceback that
    # looks like a graph fault rather than "the producer has not written this yet".
    try:
        if name not in e.attrs:
            return None
        return list(e.attrs[name].value)
    except Exception:
        return None

seen = []

def sample():
    e, where = find_rt_edge()
    if e is None:
        print("  [wait]", where); return

    ts   = attr(e, "rt_timestamps")
    head = e.attrs["rt_head_index"] if "rt_head_index" in e.attrs else None
    twl  = attr(e, "rt_twist_linear")
    twa  = attr(e, "rt_twist_angular")
    tr   = attr(e, "rt_translation")
    leg  = attr(e, "rt_translation_velocity")

    if twl is None:
        print("  [wait] edge %s carries NO rt_twist_linear yet — is the RUNNING room_concept the "
              "rebuilt binary? (ps -o lstart= vs the bin mtime)" % where)
        return

    ok_len = (len(twl) == BLOCK * HIST) and (len(twa) == BLOCK * HIST)
    print("  edge %s | head=%s" % (where, None if head is None else head.value))
    print("    rt_twist_linear  %2d floats  %s" % (len(twl), "RINGED ✓" if ok_len else "★NOT A RING"))
    print("    rt_twist_angular %2d floats" % len(twa))
    print("    rt_translation   %2d floats  rt_timestamps %d" % (len(tr or []), len(ts or [])))

    if ts:
        for i in range(min(HIST, len(ts))):
            s = slice(i * BLOCK, i * BLOCK + BLOCK)
            print("      slot %d  t=%-14d v=[% .4f % .4f % .4f]  w_z=% .4f" %
                  (i, ts[i], *twl[s], twa[i * BLOCK + 2]))
        # A slot whose timestamp is 0 is UNUSED, not "stationary at the epoch" — the two look the
        # same in the numbers and mean opposite things to a consumer.
        unused = sum(1 for t in ts[:HIST] if t == 0)
        if unused:
            print("      (%d slot(s) still unused — the ring fills over the first %d publishes)"
                  % (unused, HIST))

    if leg is not None and len(leg) >= 2:
        adv, side = leg[0], leg[1]
        slot = (head.value // BLOCK - 1) % HIST if head is not None else 0
        vx, vy = twl[slot * BLOCK], twl[slot * BLOCK + 1]
        swapped = abs(vx - side) < 1e-6 and abs(vy - adv) < 1e-6
        same    = abs(vx - adv)  < 1e-6 and abs(vy - side) < 1e-6
        verdict = ("AXIS ORDER ✓ (legacy [adv,side] appears swapped, as designed)" if swapped
                   else "★ARRAY ORDER LEAKED INTO AN AXIS-ORDER ATTRIBUTE" if same
                   else "inconclusive — the robot is probably still (adv≈side≈0)")
        print("    legacy [adv,side]=[% .4f % .4f]  vs ring slot %d [vx,vy]=[% .4f % .4f]"
              % (adv, side, slot, vx, vy))
        print("    -> %s" % verdict)

    seen.append(tuple(ts[:HIST]) if ts else ())

ticks = {"n": 0}
def tick():
    ticks["n"] += 1
    print("[t+%ds]" % ticks["n"]); sample()
    if ticks["n"] >= a.seconds:
        # The ring must ADVANCE, not just exist: a stamp set that never changes is a producer writing
        # one slot for ever, which is the pre-2026-09-09 behaviour wearing the new attribute's name.
        distinct = len(set(seen))
        print("\n%d distinct timestamp sets over %d samples -> ring %s"
              % (distinct, len(seen), "ADVANCING ✓" if distinct > 1 else "★FROZEN (or robot parked)"))
        app.quit()

QTimer.singleShot(500, sample)          # one immediate read
t = QTimer(); t.timeout.connect(tick); t.start(1000)
sys.exit(app.exec())
