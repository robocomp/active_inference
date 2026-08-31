#!/usr/bin/env python3
"""
lap_metrics — per-lap TIME and DEVIATION from the controller's own per-cycle log.

THE OBJECTIVE THESE TWO SERVE: complete the route in minimal time with minimal deviation from
it. Everything else here is context for WHY a lap was slow or loose, not a score.

★LAPS ARE FOUND FROM THE DATA, not from a lap counter. Since laps became restarts of one curve
(route_follower.cpp), a lap boundary IS a backward step in track_s — the rewind. That marker is
in the log already, needs no new plumbing, and cannot disagree with what the follower did.

★TIME IS WALL TIME BETWEEN REWINDS, deviation is the tracker's own signed cross-track over the
same rows. Both are read from the SAME cycles, so a lap cannot be fast on one clock and slow on
another — which is the mistake that makes two runs look comparable when they are not.

★THE LAST LAP IS USUALLY PARTIAL and is marked so. Averaging a partial lap into a per-lap mean
is how a run that was stopped early comes out looking quick.

Usage:  tools/lap_metrics.py [tracker_diag.csv] [--v-cap 0.70]
"""
import math, sys, statistics

def main():
    path = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "tracker_diag.csv"
    v_cap = 0.70
    if "--v-cap" in sys.argv: v_cap = float(sys.argv[sys.argv.index("--v-cap") + 1])

    rows = [l.rstrip("\n").split(",") for l in open(path) if not l.startswith("#")]
    hdr = {k: i for i, k in enumerate(rows[0])}
    D = [r for r in rows[1:] if len(r) == len(hdr)]
    def g(r, k):
        try: return float(r[hdr[k]])
        except Exception: return float("nan")

    # Split on the rewind: track_s stepping backwards by more than the projection's own slack.
    # ★AND LANDING NEAR THE LAP START. A backward step alone is not a lap: the tracker also jumps
    # backwards when it RE-ACQUIRES mid-route (a repair, or the resume bug fixed in 2222544 — measured
    # landing at 17.90 m on a 35.6 m route). Counting those as laps splits one lap into three and makes
    # every per-lap number smaller and faster than the truth. A real rewind returns to the lap start,
    # which is the low end of the arc length this run ever reached.
    s_all = [g(r, "track_s") for r in D if not math.isnan(g(r, "track_s"))]
    s_max = max(s_all) if s_all else 0.0
    near_start = 0.25 * s_max          # generous: the run-in is a few percent of a lap
    laps, cur, rejected = [], [], 0
    for a, b in zip(D, D[1:]):
        cur.append(a)
        if g(b, "track_s") < g(a, "track_s") - 1.0:
            if g(b, "track_s") <= near_start:
                laps.append(cur); cur = []
            else:
                rejected += 1
    cur.append(D[-1]); laps.append(cur)

    # ★A LAP MUST COVER THE ROUTE, and without this test the very start of a run invents one. The
    # boundary test above compares against 0.25 * the largest arc length SEEN SO FAR — and early in a
    # run that maximum is a metre or two, so any re-acquisition near the start (the tracker's first
    # projection, a route re-authoring) lands under the threshold and reads as a completed lap. Observed:
    # a "lap" of 4.9 s and 1.2 m on a 35 m route, which then drags the per-lap mean down and makes a
    # partial run look fast. So a segment that spans much less arc than the longest one is not a lap; it
    # is reported rather than silently dropped, because a run full of them means something else is wrong.
    def arc_span(L):
        v = [g(r, "track_s") for r in L if not math.isnan(g(r, "track_s"))]
        return (max(v) - min(v)) if v else 0.0
    if len(laps) > 1:
        longest = max(arc_span(L) for L in laps)
        keep = [L for L in laps if arc_span(L) >= 0.5 * longest]
        dropped = len(laps) - len(keep)
        if dropped:
            spans = [f"{arc_span(L):.1f} m" for L in laps if arc_span(L) < 0.5 * longest]
            print(f"  ⚠ {dropped} segment(s) spanning only {', '.join(spans)} against a {longest:.1f} m "
                  f"lap — too short to be laps (a re-acquisition near the start reads as a boundary). "
                  f"NOT counted.")
        laps = keep

    print(f"  {len(D)} cycles, {len(laps)} lap segment(s) (the last is partial unless it rewound)")
    if rejected:
        print(f"  ⚠ {rejected} backward step(s) in track_s did NOT return to the lap start — those are"
              f" mid-route\n    RE-ACQUISITIONS, not laps, and are deliberately not counted as boundaries.")
    print()
    # ★arc vs driven IS A DIAGNOSTIC, not bookkeeping. arc is how far the tracker's projection moved
    # along the route; driven is how far the robot actually went. On a curve they agree. When arc runs
    # far ahead of driven, the projection is ADVANCING WITHOUT THE ROBOT — the fold-crossing jumps that
    # cross-track cannot see. A lap whose arc is 35 m and whose driven distance is 20 m did not take a
    # shortcut; its arc length teleported 15 m.
    print("  lap |  time s |  arc m | driven m | drv/arc | cross rms | cross max | e_psi rms | mean v | stopped | rot@cap")
    print("  ----+---------+--------+----------+---------+-----------+-----------+-----------+--------+---------+--------")
    times, rms_all = [], []
    for i, L in enumerate(laps):
        if len(L) < 5: continue
        t = (g(L[-1], "t_ms") - g(L[0], "t_ms")) / 1000.0
        # ── DISTANCE FROM THE WHEELS, NOT FROM THE POSE ──────────────────────────────────────────
        # ★FIRST ATTEMPT WAS WRONG AND THE CROSS-CHECK CAUGHT IT. It summed pose steps while DISCARDING
        # any cycle whose implied speed exceeded the base's cap, on the grounds that such a step is the
        # estimate jumping. That filter threw away 43% of the real distance — the robot genuinely drives
        # near 0.70 m/s, so at 50 ms a large share of honest cycles cross the cap on estimator noise
        # alone. It reported 19.8 m for a lap where the raw pose sum said 34.5 and the odometry said
        # 34.7, and it turned that into a fabricated finding: "the arc advances twice as fast as the
        # robot". It does not. arc 35.1 / driven 34.7 on the same lap.
        # ★So distance comes from meas_speed, which is derived from the WHEELS and is independent of the
        # pose estimate — immune to a localiser jump by construction rather than by a filter. The raw
        # pose sum is kept beside it: when the two disagree, THAT is the localiser, and it is visible
        # instead of silently corrected.
        dist = 0.0; pose_dist = 0.0
        for a, b in zip(L, L[1:]):
            dt = (g(b, "t_ms") - g(a, "t_ms")) / 1000.0
            if not (0 < dt < 1): continue
            va = g(a, "meas_speed")
            if not math.isnan(va): dist += va * dt
            dx, dy = g(b, "pose_x") - g(a, "pose_x"), g(b, "pose_y") - g(a, "pose_y")
            if not math.isnan(dx): pose_dist += math.hypot(dx, dy)
        e  = [g(r, "pd_cross_err_m") for r in L if not math.isnan(g(r, "pd_cross_err_m"))]
        ep = [abs(g(r, "carrot_bear")) for r in L if not math.isnan(g(r, "carrot_bear"))]
        v  = [g(r, "meas_speed") for r in L if not math.isnan(g(r, "meas_speed"))]
        adv= [g(r, "cmd_adv") for r in L if not math.isnan(g(r, "cmd_adv"))]
        rot= [abs(g(r, "cmd_rot")) for r in L if not math.isnan(g(r, "cmd_rot"))]
        rms = math.sqrt(sum(x * x for x in e) / len(e)) if e else float("nan")
        mx  = max(abs(x) for x in e) if e else float("nan")
        eps = math.sqrt(sum(x * x for x in ep) / len(ep)) if ep else float("nan")
        stopped = 100.0 * sum(1 for x in adv if x <= 1e-4) / max(1, len(adv))
        atcap   = 100.0 * sum(1 for x in rot if x >= 0.79) / max(1, len(rot))
        partial = " (partial)" if i == len(laps) - 1 else ""
        sv = [g(r, "track_s") for r in L if not math.isnan(g(r, "track_s"))]
        arc = (max(sv) - min(sv)) if sv else float("nan")
        # driven / arc: 1.0 is a lap driven exactly as long as the route. ABOVE 1 means the robot
        # covered more ground than the route asked for — wandering, recovering, or turning in place.
        ratio = dist / arc if arc > 0.5 else float("nan")
        print(f"  {i+1:3d} | {t:7.1f} | {arc:6.1f} | {dist:8.1f} | {ratio:7.2f} | {rms*1000:8.0f}mm | "
              f"{mx*1000:8.0f}mm | {eps:9.3f} | {statistics.mean(v):6.2f} | {stopped:6.1f}% | "
              f"{atcap:5.1f}%{partial}")
        if not partial:
            times.append(t); rms_all.append(rms)
    if len(times) >= 2:
        print(f"\n  COMPLETE LAPS ONLY ({len(times)}):")
        print(f"    time      mean {statistics.mean(times):6.1f} s   spread {max(times)-min(times):5.1f} s "
              f"({100*statistics.pstdev(times)/max(1e-9,statistics.mean(times)):.1f}% cv)")
        print(f"    cross rms mean {statistics.mean(rms_all)*1000:6.0f} mm  spread "
              f"{(max(rms_all)-min(rms_all))*1000:5.0f} mm")
        print( "    -> the SPREAD is the repeatability signal: every lap is now the same curve,")
        print( "       so lap-to-lap variation is the robot and the world, not the route.")
    elif len(times) == 1:
        print(f"\n  only ONE complete lap ({times[0]:.1f} s, {rms_all[0]*1000:.0f} mm rms) — no spread to report.")
    else:
        print("\n  NO complete lap in this log: nothing rewound, so no lap boundary was reached.")

if __name__ == "__main__":
    main()
