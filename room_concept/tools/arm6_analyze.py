#!/usr/bin/env python3
"""Arm 6: does a longer episode buy information, or does linearisation break first?

usage: arm6_analyze.py etc/runs/arm6_base.csv etc/runs/arm6_long.csv

★★★ IT REPORTS THE VALUE COMPARISON BEFORE THE SIGMA COMPARISON, and that ordering is the
pre-registration, not a preference (EXPERIMENT.md §12.2). Both legs observe the SAME robot on the
SAME route, so k_omega's true value is identical and a longer episode should buy PRECISION ALONE:

    sigma falls, value agrees within sigma  -> the trigger was discarding information; free win
    sigma falls, VALUE SHIFTS               -> LINEARISATION BREAKING. A shift on an unchanged
                                               robot is a BIAS, and precision bought with a bias is
                                               worse than no improvement.
    neither moves                           -> the trigger was not the binding constraint; look at
                                               theta_var, whose growth may be cancelling the gain.

Reporting sigma first invites reading the arm as a success, which is why it is reported second.
"""
import csv
import math
import sys

import numpy as np


def load(path):
    d = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            for k in ('ts_ms', 'gt_x', 'gt_y', 'gt_theta', 'calib_k_w', 'calib_sig_kw',
                      'calib_informed', 'calib_cond'):
                try:
                    d.setdefault(k, []).append(float(r[k]))
                except (KeyError, TypeError, ValueError):
                    d.setdefault(k, []).append(math.nan)
    a = {k: np.asarray(v, float) for k, v in d.items()}
    # cumulative GT rotation — the yardstick for this arm, since the endpoint is information per
    # RADIAN TURNED. Distance is reported too, but it is not what the legs must match on.
    th = np.unwrap(np.nan_to_num(a['gt_theta']))
    a['rot'] = np.r_[0.0, np.cumsum(np.abs(np.diff(th)))]
    a['s'] = np.r_[0.0, np.cumsum(np.hypot(np.diff(a['gt_x']), np.diff(a['gt_y'])))]
    return a


def episodes(path):
    """The calibrator's own episode log — the manipulation check lives here."""
    th, dur = [], []
    try:
        for line in open(path):
            if line.startswith('E'):
                v = line.strip().split(',')[1:]
                try:
                    th.append(abs(float(v[2]))); dur.append(float(v[3]))
                except (IndexError, ValueError):
                    pass
    except OSError:
        return None, None
    return np.asarray(th), np.asarray(dur)


def at_rotation(a, target):
    """k_omega and its sigma once `target` radians have been turned — matched comparison."""
    i = np.searchsorted(a['rot'], target)
    if i >= len(a['rot']):
        return math.nan, math.nan
    return a['calib_k_w'][i], a['calib_sig_kw'][i]


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    A, B = load(sys.argv[1]), load(sys.argv[2])
    epA = episodes(sys.argv[1].replace('.csv', '_episodes.csv'))
    epB = episodes(sys.argv[2].replace('.csv', '_episodes.csv'))
    print(__doc__.split('usage:')[0].strip())

    print("\n── 0. MANIPULATION CHECK — did the trigger actually bind? ─────────────────────────")
    print("  %-10s %10s %10s %14s %12s" % ("leg", "dist_m", "rot_rad", "episodes", "med d_theta"))
    for name, a, ep in (("6-base", A, epA), ("6-long", B, epB)):
        th, _ = ep
        print("  %-10s %10.1f %10.1f %14s %12s"
              % (name, a['s'][-1], a['rot'][-1],
                 len(th) if th is not None else "no log",
                 "%.3f" % np.median(th) if th is not None and th.size else "-"))
    if epA[0] is not None and epB[0] is not None and epA[0].size and epB[0].size:
        r = np.median(epB[0]) / max(np.median(epA[0]), 1e-9)
        print("  median d_theta ratio long/base = %.2fx" % r)
        if r < 1.5:
            print("  ⚠ THE TRIGGER DID NOT BIND. 6-long's episodes are not materially longer, so this")
            print("    arm has not manipulated what it claims to and the endpoint below is void.")
    rmatch = min(A['rot'][-1], B['rot'][-1])
    dr = abs(A['rot'][-1] - B['rot'][-1]) / max(rmatch, 1e-9)
    print("  rotation content differs by %.1f%% -> comparisons made at %.1f rad, the shared span."
          % (100 * dr, rmatch))
    if dr > 0.25:
        print("  ⚠ >25%% apart. The endpoint is per RADIAN TURNED; this large a mismatch makes the")
        print("    comparison partly against the route. Report it, do not correct for it.")

    print("\n── 1. THE VALUE — reported FIRST, per the pre-registration ────────────────────────")
    kA, sA = at_rotation(A, rmatch)
    kB, sB = at_rotation(B, rmatch)
    print("  6-base  k_omega = %.5f  +/- %.5f" % (kA, sA))
    print("  6-long  k_omega = %.5f  +/- %.5f" % (kB, sB))
    if all(map(np.isfinite, (kA, sA, kB, sB))):
        sep = abs(kB - kA) / max(math.hypot(sA, sB), 1e-12)
        print("  the two values are %.2f combined sigmas apart" % sep)
        if sep > 2.0:
            print("  ★★★ THE VALUE MOVED. On the same robot and route the truth is identical, so this")
            print("      is a BIAS — the signature of the Jacobian's linearisation failing over a")
            print("      longer accumulated motion. Any sigma improvement below is bought with it.")
        else:
            print("  ✓ the value is unchanged within its own uncertainty; a sigma change below is")
            print("    a genuine precision gain, not a bias.")

    print("\n── 2. THE SIGMA — only meaningful if the value held ───────────────────────────────")
    if np.isfinite(sA) and np.isfinite(sB) and sB > 0:
        print("  sigma(k_omega) at %.1f rad:  base %.5f -> long %.5f  = %.2fx tighter"
              % (rmatch, sA, sB, sA / sB))
        print("  ★ CEILING: a 4x trigger is 4x more information on the episodes it governs, but")
        print("    only ~52%% of episodes are trigger-closed (the rest end on a correction's falling")
        print("    edge, which no knob controls), so the whole-window ceiling is ~2.6x and sigma")
        print("    falls by at most sqrt(2.6) = 1.6x. ABOVE that is not this mechanism.")

    print("\n── 3. Whole-leg endpoints, for the record ─────────────────────────────────────────")
    print("  %-10s %12s %12s %10s" % ("leg", "k_omega", "sigma", "cond"))
    for name, a in (("6-base", A), ("6-long", B)):
        c = a['calib_cond'][np.isfinite(a['calib_cond'])]
        print("  %-10s %12.5f %12.5f %10s"
              % (name, a['calib_k_w'][-1], a['calib_sig_kw'][-1],
                 "%.2f" % c[-1] if c.size else "-"))
    print("\n  Run tools/excitation.py on each leg's *_episodes.csv for the heading block's")
    print("  lambda_min and the per-parameter data share.")


if __name__ == '__main__':
    main()
