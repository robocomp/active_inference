#!/usr/bin/env python3
"""reversal_map.py — WHERE on the route the rotation reverses.

The reversal count is the loudest defect in this stack (~100/lap against a 38.8 baseline) and until now
it was a scalar: a run could say reversals HAPPENED but not where, so every claim about their cause was
an inference. This reads the actuation stream's `route_s_m` column and bins reversals by ARC LENGTH,
folded onto one lap, so repeated offenders stand out from one-off events.

  scripts/reversal_map.py "etc/runs/complete tour/<run>_profile.csv" [route_geometry.csv]

With route_geometry.csv it also prints the (x,y) of each hotspot, which is what makes a hotspot
actionable: a place is either an authored waypoint (only the tour can fix it) or it is not.

★The deadband MUST match TrajectoryStats' (0.05 rad/s). A different one measures a different quantity
and the number stops being comparable with every run already recorded.
"""
import csv
import math
import sys
from collections import defaultdict

DEADBAND = 0.05      # rad/s — same as MissionRunner::sample, do not "improve"
BIN_M = 0.25


def load_profile(path):
    rows = []
    with open(path) as f:
        # Profile CSVs carry TWO '#' comment lines before the header.
        lines = [l for l in f if not l.startswith("#")]
    for r in csv.DictReader(lines):
        try:
            rows.append((float(r["t_ms"]), int(r["lap"]), float(r["rot_rps"]),
                         float(r.get("route_s_m", -1))))
        except (KeyError, ValueError):
            continue
    return rows


def load_route(path):
    """Smoothed route samples, in order, from route_geometry.csv (last build wins)."""
    ev = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["kind"] == "smoothed":
                ev[int(r["event_id"])].append((float(r["x"]), float(r["y"])))
    return ev[max(ev)] if ev else []


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    rows = load_profile(sys.argv[1])
    route = load_route(sys.argv[2]) if len(sys.argv) > 2 else []
    if not rows:
        print("no usable rows")
        return 1
    if all(s < 0 for _, _, _, s in rows):
        print("route_s_m is -1 in every row: this run did not drive a continuous route, or it predates "
              "the column. Nothing to map.")
        return 1

    laps = max(l for _, l, _, _ in rows)
    s_max = max(s for _, _, _, s in rows)
    lap_len = s_max / laps if laps else s_max
    print(f"{len(rows)} samples, {laps} lap(s), route {s_max:.1f} m total, {lap_len:.1f} m per lap")

    reversals = []            # (s_absolute, lap)
    prev_sign = 0
    for _, lap, rot, s in rows:
        sign = 1 if rot > DEADBAND else (-1 if rot < -DEADBAND else 0)
        if sign == 0:
            continue
        if prev_sign != 0 and sign != prev_sign and s >= 0:
            reversals.append((s, lap))
        prev_sign = sign
    print(f"{len(reversals)} reversals ({len(reversals)/max(1,laps):.0f} per lap), deadband {DEADBAND}")

    # Folded onto ONE lap: a place that offends on every lap is route geometry, a place that offends once
    # is an event. That distinction is the whole reason for folding rather than plotting raw arc length.
    bins = defaultdict(list)
    for s, lap in reversals:
        bins[int((s % lap_len) / BIN_M)].append(lap)
    print(f"\nhotspots (folded onto one lap, {BIN_M} m bins, worst first):")
    print(f"{'s_lap':>8} {'count':>6} {'laps':>10}   position")
    for b, hits in sorted(bins.items(), key=lambda kv: -len(kv[1]))[:15]:
        s_lap = (b + 0.5) * BIN_M
        pos = ""
        if route:
            i = min(len(route) - 1, int(s_lap / 0.05))
            pos = f"({route[i][0]:+.2f},{route[i][1]:+.2f})"
        laps_hit = "".join(str(l) for l in sorted(set(hits)))
        print(f"{s_lap:8.2f} {len(hits):6d} {laps_hit:>10}   {pos}")

    # How concentrated is it? If the top few bins hold most of the reversals, the cause is a PLACE; if the
    # distribution is flat, it is a property of the controller and no amount of route work will fix it.
    counts = sorted((len(v) for v in bins.values()), reverse=True)
    tot = sum(counts)
    for k in (5, 10, 20):
        if len(counts) >= k:
            print(f"top {k:2d} bins hold {100*sum(counts[:k])/tot:5.1f}% of reversals "
                  f"({k*BIN_M:.1f} m of a {lap_len:.1f} m lap)")
    print(f"occupied bins: {len(bins)} of {int(lap_len/BIN_M)} ({100*len(bins)/max(1,int(lap_len/BIN_M)):.0f}% "
          f"of the lap sees at least one reversal)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
