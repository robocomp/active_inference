#!/usr/bin/env python3
"""crossing_window_mock.py -- seconds-fast model of WHEN room_concept localises on odometry only and keeps its relocaliser
off across one door crossing (Fable review 2026-10-31, B1/B2).

CAVEAT: a model of the KEYING (which graph facts room_concept reads, under DDS delays), not of the localiser itself.

World (truth, per tick): the body goes APPROACH -> DOORWAY (its body overlaps the door plane) -> PAST (at P_post, 1.3 m in)
and the new room is ADOPTED by room_concept some ticks after its switch (a recall's placement; a new room at once).
DANGER = the body is in the doorway, or the robot is in the new room but room_concept has not adopted it: an SDF fit or a
relocaliser run then sees one room through the other's door. Invariant: in DANGER, odometry-only ON and relocaliser OFF.

Graph writes (each its own DDS message, random visibility delay, so an insert and a delete can arrive in either order):
  controller : `executing` on aff_<door>_cross at cross start (current) / link 0 `interacting` robot->door (fixed);
               link 1 at P_post; (fixed) link 0 removed once it SEES room_concept's switch (link 2).
  door_concept: may WITHDRAW the cross affordance mid-cross (deletes the node, its `executing` edge dies) -- live behaviour.
               link 3 on seeing link 2.
  ltsm       : link 4 (archive) on seeing link 3; then door_concept deletes OLD_FACE (links 1, 2 die with it).
room_concept: switch on seeing link 1 (link 2); adopts after `adopt` ticks.
  current: crossing = `executing` visible on a door of its room; relocaliser off while crossing or link 2 lives.
  fixed  : crossing = link 0 or link 1 visible on a door of its room; after its switch, relocaliser off and odometry-only
           until it has adopted the room.
Usage: python3 crossing_window_mock.py [--seeds 2000] [--dmax 4] [--pwithdraw 0.5]
"""
import argparse, random


def run(seed, mode, dmax, pwithdraw, lead):
    rng = random.Random(seed)
    t_cross = 5
    t_door0 = t_cross + lead + rng.randint(0, 4)  # body reaches the door plane (lead = cross start at the vantage -> plane)
    t_door1 = t_door0 + rng.randint(3, 8)        # body clear of the plane
    t_post = t_door1 + rng.randint(1, 4)         # body at P_post: link 1
    t_withdraw = rng.randint(t_cross + 1, t_post) if rng.random() < pwithdraw else None
    adopt_delay = rng.randint(0, 6)              # recall placement (0 = a new room)
    writes = []                                  # (visible_at, key, value)

    def put(t, key, val): writes.append((t + rng.randint(0, dmax), key, val))
    # controller's cross start
    put(t_cross, 'executing' if mode == 'current' else 'link0', True)
    if t_withdraw is not None: put(t_withdraw, 'executing', False)   # door_concept withdraws: its node and edge die
    put(t_post, 'link1', True)
    view = {}
    switched_at = adopted_at = None
    link2_seen_ct = link3_at = link4_at = None
    for t in range(0, t_post + 60):
        for (ta, k, v) in [w for w in writes if w[0] == t]: view[k] = v
        # room_concept
        if switched_at is None and view.get('link1'):
            switched_at = t; adopted_at = t + adopt_delay; put(t, 'link2', True)
        # controller (fixed): drop link 0 once it SEES the switch
        if mode == 'fixed' and view.get('link2') and link2_seen_ct is None:
            link2_seen_ct = t; put(t, 'link0', False)
        # door_concept: link 3; ltsm: link 4; then OLD_FACE deleted
        if view.get('link2') and link3_at is None: link3_at = t; put(t, 'link3', True)
        if view.get('link3') and link4_at is None:
            link4_at = t; put(t + rng.randint(0, 3), 'link2', False); put(t + rng.randint(0, 3), 'link1', False)
        in_new = switched_at is not None
        if mode == 'current':
            crossing = (not in_new) and view.get('executing', False)
            reloc_off = crossing or view.get('link2', False)
            odom_only = crossing
        else:
            crossing = (not in_new) and (view.get('link0', False) or view.get('link1', False))
            hold = in_new and t < adopted_at
            reloc_off = crossing or hold
            odom_only = crossing or hold
        in_door = t_door0 <= t < t_door1
        unadopted = t >= t_post and (adopted_at is None or t < adopted_at)
        if (in_door or unadopted) and not (odom_only and reloc_off):
            return f"t={t} body {'in the doorway' if in_door else 'in the new room, not adopted'}: odom_only={odom_only} reloc_off={reloc_off}" \
                   f"{' (cross withdrawn at t=' + str(t_withdraw) + ')' if t_withdraw else ''}"
    return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', type=int, default=2000)
    ap.add_argument('--dmax', type=int, default=4); ap.add_argument('--pwithdraw', type=float, default=0.5)
    ap.add_argument('--lead', type=int, default=2, help='ticks from cross start to the body at the door plane')
    a = ap.parse_args()
    for mode in ('current', 'fixed'):
        fails = [(s, r) for s in range(a.seeds) if (r := run(s, mode, a.dmax, a.pwithdraw, a.lead))]
        print(f"{mode:8s}: {len(fails)}/{a.seeds} seeds violate the invariant" + (f"; first: seed {fails[0][0]}: {fails[0][1]}" if fails else ""))


if __name__ == '__main__':
    main()
