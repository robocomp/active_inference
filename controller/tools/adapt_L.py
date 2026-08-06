#!/usr/bin/env python3
"""Propose the next plain_L from the run history: minimise cross-track rms subject to a hard
   constraint on rotational effort.

WHY THIS SHAPE. Measured over nine laps on this tour (three 3-lap runs, identical config):

    cross_track_rms       cv  2.3%      <- the objective
    rot_effort per metre  cv  2.3%      <- the constraint
    dev_norm              cv  2.3%
    smooth_rot            cv 23.2%      x
    smooth_lin            cv 31.8%      x
    jerk per metre        cv 42.6%      x
    min_clearance         cv  132%      x

Only the first three repeat. The smoothness terms are properties of the ROUTE, which is re-planned
against a live grid every run — one 119 mm waypoint repair moved jerk/m by 38% between two runs with
byte-identical authored waypoints. So they cannot be optimised against on hardware, and neither can J,
which is built from them.

WHY A CONSTRAINT AND NOT A WEIGHTED SUM. Minimising rms alone drives the gains 2/L and 1/L^2 up until
the loop rings — the policy would find the stability boundary, not an optimum. What tightening costs
shows up elsewhere (TV(w)/m, lap time), so the hard limit is what puts that cost back in, and it keeps
"how much roughness is acceptable" an explicit engineering choice rather than a weight to justify.

⚠THE "rms FALLS MONOTONICALLY WITH L" CLAIM IS WITHDRAWN (2026-08-06). It came from tracker_sim back
when that bench carried a hand-written REPLICA of the tracker; the bench now links PlainTracker itself,
and the same sweep is NOT monotone:

    L      0.45    0.50    0.60    0.75    1.00
    rms mm 75.1    98.0    90.7   108.3   236.1
    rot/m  1.293   1.314   1.185   1.117   1.137

So there may well be an interior minimum, and the step-down search below can walk past one. It is still
sound as a bracketing search under the constraint — it just no longer has a monotonicity argument
behind it. Treat a reversal as information, not noise, and halve --step.

⚠THE SIM IS A PRIOR, NOT AN ANSWER. It has no pose noise, and low L means high gain on a noisy
measurement, so the robot suffers where the bench does not. Its rms-optimal L=0.25 also sits below the
~0.45 lower bound the loop-margin rule gives. Use it to order candidates; let the robot decide.

USAGE
    tools/adapt_L.py [mission_metrics.csv] [--budget 0.87] [--floor 0.45] [--step 0.10] [--noise 0.05]

    --budget  rad/m ceiling on rot_effort/distance. Default 0.87 = just above the highest observed
              today (0.850 / 0.863 / 0.818), i.e. "never rougher than it is now".
    --floor   L below which the loop-margin rule says the design degrades. A hard stop.
    --step    initial search step, halved whenever the search reverses direction.
    --noise   fractional improvement that counts as real. 0.05 = 5%, about 2 sigma at the measured
              2.3% spread, so a single run can resolve it.

Python rather than C++ on purpose: float() here is locale-independent, while these machines run
es_ES where the C library's strtof silently truncates "0.11" to 0 (see CLAUDE.md).
"""
import csv
import sys
from collections import defaultdict


def main() -> int:
    args = sys.argv[1:]
    path = next((a for a in args if not a.startswith("--")), "mission_metrics.csv")

    def opt(name: str, default: float) -> float:
        if f"--{name}" in args:
            return float(args[args.index(f"--{name}") + 1])
        return default

    budget, floor = opt("budget", 0.87), opt("floor", 0.45)
    step, noise = opt("step", 0.10), opt("noise", 0.05)

    try:
        rows = list(csv.DictReader(open(path)))
    except FileNotFoundError:
        print(f"no such file: {path}")
        return 1
    if not rows or "plain_L" not in rows[0]:
        print(f"'{path}' has no plain_L column — it predates per-run gain logging.\n"
              f"Runs recorded before that cannot be attributed to a gain and are unusable here.")
        return 1

    # One record per run: the gain, the objective, and the constrained quantity.
    runs = []
    for r in rows:
        try:
            if r.get("control_mode") != "plain" or int(r["laps"]) < 1:
                continue
            dist = float(r["distance_m"])
            if dist < 1.0:
                continue
            runs.append((float(r["plain_L"]), float(r["cross_track_rms_m"]),
                         float(r["rot_effort_rad"]) / dist, int(r["laps"])))
        except (ValueError, KeyError):
            continue
    if not runs:
        print("no usable 'plain' runs with a recorded L.")
        return 1

    by_L = defaultdict(list)
    for L, rms, rot, laps in runs:
        by_L[round(L, 3)].append((rms, rot, laps))

    print(f"  budget {budget:.3f} rad/m   floor L {floor:.2f}   step {step:.2f}   "
          f"noise {100*noise:.0f}%   ({len(runs)} runs)\n")
    print(f"  {'L':>6} {'n':>3} {'laps':>5} {'rms m':>9} {'rot/m':>8}   verdict")
    table = []
    for L in sorted(by_L):
        rs = [x[0] for x in by_L[L]]
        ro = [x[1] for x in by_L[L]]
        lp = sum(x[2] for x in by_L[L])
        rms, rot = sum(rs) / len(rs), sum(ro) / len(ro)
        ok = rot <= budget
        table.append((L, rms, rot, ok, len(rs)))
        print(f"  {L:>6.2f} {len(rs):>3} {lp:>5} {rms:>9.4f} {rot:>8.3f}   "
              f"{'feasible' if ok else 'OVER BUDGET'}")

    feasible = [t for t in table if t[3]]
    if not feasible:
        worst = min(table, key=lambda t: t[2])
        nxt = round(worst[0] + step, 3)
        print(f"\n  every tested L exceeds the budget. Loosen L to reduce rotational effort.")
        print(f"  NEXT: plain_L = {nxt:.2f}  (from {worst[0]:.2f}, the least-rough tested)")
        return 0

    best = min(feasible, key=lambda t: t[1])
    tested = sorted(t[0] for t in table)
    print(f"\n  best feasible: L = {best[0]:.2f}  rms {best[1]:.4f} m  rot/m {best[2]:.3f}"
          f"  ({best[4]} run(s))")

    # Pattern search. rms falls monotonically with L, so the useful direction is DOWN until either the
    # constraint binds or the floor is reached; the constraint is what stops it, by design.
    lower = [t for t in table if t[0] < best[0]]
    infeasible_below = [t for t in lower if not t[3]]
    if infeasible_below:
        # Bracketed: feasible above, over-budget below. Bisect toward the boundary.
        hi = best[0]
        lo = max(t[0] for t in infeasible_below)
        nxt = round((hi + lo) / 2.0, 3)
        why = f"bracketed between {lo:.2f} (over budget) and {hi:.2f} (feasible) — bisecting"
    else:
        nxt = round(best[0] - step, 3)
        why = f"constraint not binding at {best[0]:.2f} — stepping down to buy rms"

    if nxt < floor:
        print(f"\n  NEXT: none. The search wants L = {nxt:.2f}, below the {floor:.2f} floor where the\n"
              f"        loop-margin rule says the design degrades. L = {best[0]:.2f} is the answer.")
        return 0
    if any(abs(nxt - t[0]) < 1e-3 for t in table):
        print(f"\n  NEXT: none. L = {nxt:.2f} has already been tested; halve --step to refine.")
        return 0

    print(f"\n  NEXT: plain_L = {nxt:.2f}   ({why})")
    print(f"  Set PlainTrackerL in etc/config.toml, RESTART the controller (config is read once at\n"
          f"  process start), and run 3 laps. Verify with `ps -o lstart` that the process is newer\n"
          f"  than the edit before attributing anything to it.")
    if best[4] < 2:
        print(f"  ⚠L = {best[0]:.2f} has only {best[4]} run. At 2.3% spread a single run resolves ~5%;\n"
              f"   if the next result is closer than that, repeat rather than believe it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
