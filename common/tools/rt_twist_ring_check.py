#!/usr/bin/env python3
"""rt_twist_ring_check — is the twist actually RINGED, and does slot i of it belong to slot i
of the pose?

WHY THIS EXISTS. rt_twist_linear / rt_twist_angular (cortex, 2026-09-09) replace the loose
rt_translation_velocity pair. The whole claim of the change is that a twist now lands in the SAME
ring slot as the pose it was measured with, and therefore inherits that block's timestamp. That claim
is not visible from the agent's own logs and it fails SILENTLY: a twist packed against the wrong slot
is a well-formed 3-vector attached to the wrong instant. So it gets its own check.

It also verifies the ★FRAME AND AXIS conversion, which is the other half of the change and the one
that has already cost the same class of bug twice. The ring holds the CHILD's twist in the CHILD's
own axes; on this fleet the localisation edge is anchored parent=ROBOT child=ROOM, so that is the
ROOM's apparent motion, not the robot's. The legacy pair is the ROBOT's body twist in ARRAY order
[adv, side, _]. They are related by the SE(2) adjoint of the edge's own pose, so the check is a
ROUND TRIP: convert the ring twist back to the robot's frame and it must reproduce the legacy pair.
A plain "are these the same two numbers swapped" comparison was the OLD check and is now wrong --
it was written when the producer put the robot's twist straight into the ring, which is exactly the
defect fixed on 2026-09-10 (extrapolation was then WORSE than not extrapolating).

  usage:  python3 rt_twist_ring_check.py [--agent-id 900] [--seconds 6]

★pydsr needs a Qt event loop or the graph reads back EMPTY (0 nodes) — see the QTimer below. A run
that prints "no nodes" is almost always this, not an absent graph.
"""
import argparse, math, sys
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

BLOCK = 3
# ★RING DEPTH IS PER-EDGE, NOT A CONSTANT IN THIS FILE. It used to be hard-coded to 5 while
# room_concept sets HISTORY_SIZE=25, so "NOT A RING" fired on a perfectly good 25-deep ring and the
# slot arithmetic picked the wrong block. Read it off rt_timestamps, which is the edge's own answer.
HIST = 5   # fallback only; hist_of() below is what the checks use

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
best = {}

def sample():
    e, where = find_rt_edge()
    if e is None:
        print("  [wait]", where); return

    ts   = attr(e, "rt_timestamps")
    head = e.attrs["rt_head_index"] if "rt_head_index" in e.attrs else None
    twl  = attr(e, "rt_twist_linear")
    twa  = attr(e, "rt_twist_angular")
    tr   = attr(e, "rt_translation")
    rot  = attr(e, "rt_rotation_euler_xyz")
    leg  = attr(e, "rt_translation_velocity")

    if twl is None:
        print("  [wait] edge %s carries NO rt_twist_linear yet — is the RUNNING room_concept the "
              "rebuilt binary? (ps -o lstart= vs the bin mtime)" % where)
        return

    hist = len(ts) if ts else HIST
    ok_len = (len(twl) == BLOCK * hist) and (len(twa) == BLOCK * hist)
    print("  edge %s | head=%s" % (where, None if head is None else head.value))
    print("    rt_twist_linear  %2d floats  %s" % (len(twl), "RINGED ✓" if ok_len else "★NOT A RING"))
    print("    rt_twist_angular %2d floats" % len(twa))
    print("    rt_translation   %2d floats  rt_timestamps %d" % (len(tr or []), len(ts or [])))

    if ts:
        for i in range(min(hist, len(ts))):
            s = slice(i * BLOCK, i * BLOCK + BLOCK)
            print("      slot %d  t=%-14d v=[% .4f % .4f % .4f]  w_z=% .4f" %
                  (i, ts[i], *twl[s], twa[i * BLOCK + 2]))
        # A slot whose timestamp is 0 is UNUSED, not "stationary at the epoch" — the two look the
        # same in the numbers and mean opposite things to a consumer.
        unused = sum(1 for t in ts[:hist] if t == 0)
        if unused:
            print("      (%d slot(s) still unused — the ring fills over the first %d publishes)"
                  % (unused, hist))

    if leg is not None and len(leg) >= 2 and twa is not None and tr is not None and rot is not None:
        adv, side = leg[0], leg[1]
        # ★DERIVE THE SLOT FROM THE STAMPS, NEVER FROM rt_head_index MODULO A HARD-CODED HIST.
        # This file had HIST=5 while room_concept sets HISTORY_SIZE=25, so the modulo picked the
        # wrong block — the same wrong-ring-depth mistake that has already produced one bad
        # measurement in this codebase. The newest stamp is unambiguous and needs no such constant.
        # ★IT MUST BE THE NEWEST SLOT, AND ONLY THE NEWEST. The legacy pair is a LOOSE attribute:
        # one value, no ring, no stamp, overwritten every publish. So it can only be compared with
        # the block from the same instant. A previous version of this check picked the FASTEST block
        # in the ring instead, to get a conclusive answer while the robot idled -- and immediately
        # reported a 0.0035 m/s "disagreement" that was nothing but the robot's speed changing
        # between two different instants. That is not evidence about the conversion; it is two
        # measurements of different moments being subtracted. Conclusiveness comes from SAMPLING
        # LONGER (below), never from pairing across time.
        slot = max(range(len(ts)), key=lambda k: ts[k]) if ts else 0
        vx, vy = twl[slot * BLOCK], twl[slot * BLOCK + 1]
        wz = twa[slot * BLOCK + 2]
        # The stored matrix is parent<-child; xi_parent = -Ad_T(xi_child), Ad_(R,t)(v,w) = (Rv + w(ty,-tx), w).
        tx, ty = tr[slot * BLOCK], tr[slot * BLOCK + 1]
        th = rot[slot * BLOCK + 2]
        c, sn = math.cos(th), math.sin(th)
        back_x = -(c * vx - sn * vy + wz * ty)
        back_y = -(sn * vx + c * vy - wz * tx)
        # legacy is ARRAY order [adv, side]; the recovered body twist is AXIS order [lateral, forward]
        moving = (math.hypot(vx, vy) + abs(wz)) > 0.02
        err = math.hypot(back_x - side, back_y - adv)
        verdict = ("ROUND TRIP ✓ (ring -> adjoint -> robot frame reproduces the legacy pair)" if err < 1e-3
                   else "★RING AND LEGACY PAIR DISAGREE by %.4f m/s — the adjoint or the anchoring is wrong" % err)
        if not moving:
            verdict = "inconclusive at this instant — at rest every convention agrees"
        motion = math.hypot(vx, vy) + abs(wz)
        if motion > best.get("motion", 0.0):
            best.update(motion=motion, err=err, adv=adv, side=side, vx=vx, vy=vy, wz=wz,
                        bx=back_x, by=back_y)
        print("    legacy [adv,side]=[% .4f % .4f]   ring slot %d [vx,vy,wz]=[% .4f % .4f % .4f]"
              % (adv, side, slot, vx, vy, wz))
        print("    recovered body [lateral,forward]=[% .4f % .4f]   round-trip err %.5f m/s"
              % (back_x, back_y, err))
        print("    -> %s" % verdict)

    seen.append(tuple(ts) if ts else ())

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
        # ── THE VERDICT COMES FROM THE MOST-MOVING INSTANT SEEN, NOT THE LAST ONE ───────────────
        # Every per-tick line above compares the newest block with the legacy pair at that SAME
        # instant, which is the only honest pairing. Across ticks we keep whichever instant carried
        # the most motion, because that is the one able to discriminate: at rest the conversion is
        # the identity and a pass there proves nothing. Drive the robot and re-run if this says the
        # window was too still.
        if best:
            print("\nROUND-TRIP VERDICT — from the fastest INSTANT sampled (|v|+|w| = %.4f):" % best["motion"])
            print("  ring   [vx,vy,wz]        = [% .4f % .4f % .4f]" % (best["vx"], best["vy"], best["wz"]))
            print("  recovered body [lat,fwd] = [% .4f % .4f]" % (best["bx"], best["by"]))
            print("  legacy [adv,side]        = [% .4f % .4f]" % (best["adv"], best["side"]))
            if best["motion"] < 0.05:
                print("  -> TOO STILL TO DISCRIMINATE (err %.5f m/s). Drive the robot and re-run."
                      % best["err"])
            elif best["err"] < 1e-3:
                print("  -> ✓ the adjoint round trip reproduces the robot's body twist (err %.5f m/s)"
                      % best["err"])
            else:
                print("  -> ★DISAGREE by %.4f m/s — the adjoint or the anchoring is wrong" % best["err"])
        app.quit()

QTimer.singleShot(500, sample)          # one immediate read
t = QTimer(); t.timeout.connect(tick); t.start(1000)
sys.exit(app.exec())
