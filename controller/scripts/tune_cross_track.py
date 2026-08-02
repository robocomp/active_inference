#!/usr/bin/env python3
"""
tune_cross_track.py — offline gain diagnosis for the PD tracker, from a recorded lap.

WHY THIS IS NOT "MINIMISE RMS(e)"
RMS cross-track is U-shaped in the gain: too low leaves a standing offset, too high oscillates, and
both raise it. A scalar minimiser cannot tell which side of the valley it is on without probing the
robot. The two failure modes are separable because they leave DIFFERENT STRUCTURE in the same signal:

  UNDER-GAIN : e carries a persistent component correlated with path curvature. Pure pursuit's
               steady-state offset is ~ kappa*L^2/2, so e follows the SIGN of kappa and crosses zero
               rarely. Diagnostic: signed corr(e, kappa), and the slope of e on kappa (in metres per
               1/m, i.e. the effective L^2/2 the tracker is behaving with).
  OVER-GAIN  : e oscillates ABOUT zero — high zero-crossing rate, energy at the closed-loop natural
               frequency, negative autocorrelation at half a period. Uncorrelated with kappa.

Two oppositely-signed diagnostics from one channel, which is what makes the problem well posed.

AND THE PART THAT ACTUALLY TRANSFERS TO A REAL ROBOT
Gains usually need retuning across platforms because the PLANT changed — actuator lag, slip, latency
— not because the geometry did. Given commanded and measured rotation this script identifies a
first-order lag with delay, so the gain can be COMPUTED from the identified plant instead of searched
for. That transfers by construction. Stanley's form already removes the speed dependence, which is the
other large cross-platform term.

USAGE
    scripts/tune_cross_track.py <run-stamp-prefix> [...]
    scripts/tune_cross_track.py "etc/runs/complete tour/20260802-143347"
Reads <prefix>_mppi_diag.csv and <prefix>.json (for the run window). Degrades gracefully: columns it
does not find are reported as unavailable rather than silently skipped.
"""
import csv
import json
import math
import sys

KAPPA_ABSENT = -999.0          # sentinel written by the agent when there is no continuous route


def load(prefix):
    with open(prefix + ".json") as fh:
        meta = json.load(fh)
    t0 = meta["run_start_ms"]
    t1 = t0 + meta["trajectory"]["duration_s"] * 1000.0
    rows = []
    with open(prefix + "_mppi_diag.csv") as fh:
        reader = csv.DictReader(ln for ln in fh if not ln.startswith("#"))
        header = list(reader.fieldnames or [])
        for rec in reader:
            # DROP RAGGED ROWS WHOLE. A short final row (the process was killed mid-write) makes
            # DictReader emit None for the tail columns; keeping it with the key omitted produces rows
            # with different key sets, and then a column that exists in the file appears missing
            # depending on which row you happen to inspect first.
            if any(v is None for v in rec.values()) or None in rec:
                continue
            try:
                vals = {k: float(v) for k, v in rec.items()}
            except (TypeError, ValueError):
                continue
            # ALWAYS cut to the run window: these files span idle time either side of the run, and
            # uncut they have silently changed every summary statistic computed from them.
            if t0 <= vals.get("t_ms", -1) <= t1:
                rows.append(vals)
    return meta, rows, header


def stats(xs):
    n = len(xs)
    if n == 0:
        return None
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    return mean, math.sqrt(var), sorted(xs)[n // 2]


def corr(xs, ys):
    n = len(xs)
    if n < 8:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 1e-12 or syy <= 1e-12:
        return None
    return sxy / math.sqrt(sxx * syy), sxy / sxx      # (correlation, slope of y on x)


def zero_crossings(xs, deadband):
    """Sign changes that actually leave a deadband — otherwise sensor noise about zero counts as
    oscillation and every gain looks too high."""
    crossings, last = 0, 0
    for x in xs:
        s = 1 if x > deadband else (-1 if x < -deadband else 0)
        if s != 0:
            if last != 0 and s != last:
                crossings += 1
            last = s
    return crossings


def autocorr(xs, lag):
    n = len(xs) - lag
    if n < 8:
        return None
    m = sum(xs) / len(xs)
    num = sum((xs[i] - m) * (xs[i + lag] - m) for i in range(n))
    den = sum((x - m) ** 2 for x in xs)
    return num / den if den > 1e-12 else None


def identify_plant(cmd, meas, dt):
    """First-order lag with transport delay:  meas[k] = a*meas[k-1] + (1-a)*K*cmd[k-d].
    Grid-searches the delay and least-squares fits (a, (1-a)K) for each — a small enough problem that
    a closed form per delay is exact and no optimiser is needed."""
    best = None
    for d in range(0, min(12, len(cmd) // 4)):
        y, x1, x2 = [], [], []
        for k in range(max(1, d), len(meas)):
            y.append(meas[k]); x1.append(meas[k - 1]); x2.append(cmd[k - d])
        if len(y) < 20:
            continue
        # normal equations for y = a*x1 + b*x2
        s11 = sum(v * v for v in x1); s22 = sum(v * v for v in x2)
        s12 = sum(p * q for p, q in zip(x1, x2))
        sy1 = sum(p * q for p, q in zip(y, x1)); sy2 = sum(p * q for p, q in zip(y, x2))
        det = s11 * s22 - s12 * s12
        if abs(det) < 1e-9:
            continue
        a = (sy1 * s22 - sy2 * s12) / det
        b = (sy2 * s11 - sy1 * s12) / det
        resid = sum((yy - a * p - b * q) ** 2 for yy, p, q in zip(y, x1, x2))
        tot = sum(yy * yy for yy in y)
        r2 = 1.0 - resid / tot if tot > 1e-12 else 0.0
        if not (0.0 < a < 0.999):
            continue
        tau = -dt / math.log(a)
        gain = b / (1.0 - a)
        if best is None or r2 > best["r2"]:
            best = {"delay_steps": d, "delay_s": d * dt, "tau_s": tau, "gain": gain, "r2": r2}
    return best


def report(prefix):
    meta, rows, header = load(prefix)
    print(f"\n=== {prefix.split('/')[-1]} ===")
    print(f"  {len(rows)} cycles in the run window "
          f"({meta['trajectory']['duration_s']:.1f} s, {meta['trajectory']['distance_m']:.1f} m)")
    if not rows:
        print("  no rows inside the window — nothing to say"); return
    have = set(header)

    # The column was renamed pd_cross_err_m on 2026-08-02 (it is the error THE PD LAW SAW, in the robot
    # frame — not the run JSON's cross_track_rms_m, which is measured against the spline with the
    # opposite sign). Every lap recorded before the rename carries the old name; accept both rather than
    # orphaning that data.
    ekey = next((k for k in ("pd_cross_err_m", "cross_track_m") if k in have), None)
    if ekey is None:
        print("  cross-track NOT LOGGED in this run — rebuild and re-run; nothing below is computable")
        return
    e = [r[ekey] for r in rows]
    m, sd, med = stats([abs(v) for v in e])
    bias = sum(e) / len(e)
    print(f"  |e|  mean {m:.4f}  median {med:.4f}  sd {sd:.4f}   signed bias {bias:+.4f} m")

    dt = 0.1
    if len(rows) > 1:
        dts = [(rows[i + 1]["t_ms"] - rows[i]["t_ms"]) / 1000.0 for i in range(len(rows) - 1)]
        dts = [d for d in dts if 0.0 < d < 1.0]
        if dts:
            dt = sorted(dts)[len(dts) // 2]

    # ── OVER-GAIN SIDE ────────────────────────────────────────────────────────────────────────
    deadband = max(0.01, 0.15 * m)
    zc = zero_crossings(e, deadband)
    zc_hz = zc / (len(rows) * dt)
    ac = autocorr(e, max(1, int(round(0.5 / dt))))
    print(f"  oscillation: {zc} sign changes = {zc_hz:.2f} Hz"
          + (f"   autocorr@0.5s {ac:+.2f}" if ac is not None else ""))

    # ── UNDER-GAIN SIDE ───────────────────────────────────────────────────────────────────────
    if "path_kappa" not in have:
        print("  path_kappa NOT LOGGED — the under-gain half is UNAVAILABLE, so this run cannot")
        print("    distinguish 'gain too low' from 'gain about right'. Re-run with the current binary.")
    else:
        pairs = [(r["path_kappa"], r[ekey]) for r in rows
                 if r["path_kappa"] > KAPPA_ABSENT + 1.0]
        # Only curved stretches carry information about the gain: on a straight, e says nothing about
        # it. This is the persistent-excitation condition, applied by weighting rather than gating.
        curved = [(k, v) for k, v in pairs if abs(k) > 0.05]
        print(f"  curvature available on {len(pairs)}/{len(rows)} cycles, "
              f"{len(curved)} of them curved (|kappa|>0.05)")
        if len(curved) >= 20:
            c = corr([k for k, _ in curved], [v for _, v in curved])
            if c:
                r, slope = c
                print(f"  corr(e, kappa) = {r:+.3f}   slope {slope:+.4f} m per (1/m)")
                if abs(slope) > 1e-6:
                    print(f"    -> implied effective lookahead L = {math.sqrt(abs(slope) * 2):.2f} m")
                if r > 0.3:
                    print("    VERDICT: curvature-correlated offset => gain is TOO LOW (raise it)")
                elif zc_hz > 0.6:
                    print("    VERDICT: little curvature bias, high sign-change rate => gain TOO HIGH")
                else:
                    print("    VERDICT: no curvature bias, no oscillation => gain is about right")
        else:
            print("  too few curved cycles to fit the under-gain diagnostic")

    # ── PLANT IDENTIFICATION ──────────────────────────────────────────────────────────────────
    if "meas_rot" not in have:
        print("  meas_rot NOT LOGGED — plant identification unavailable (this is the part that")
        print("    actually transfers to a real robot). Re-run with the current binary.")
        return
    cmd = [r["cmd_rot"] for r in rows]
    meas = [r["meas_rot"] for r in rows]
    fit = identify_plant(cmd, meas, dt)
    if fit is None or fit["r2"] < 0.2:
        print(f"  plant fit FAILED or is not credible (r2={fit['r2']:.2f})" if fit else
              "  plant fit FAILED")
        print("    meas_rot is EMA-smoothed and differenced from a ~5 Hz pose feed, so it lags; a poor")
        print("    fit here is as likely to be the measurement as the plant. Do not tune on it.")
        return
    print(f"  plant: tau {fit['tau_s']:.3f} s, delay {fit['delay_s']:.2f} s, dc gain {fit['gain']:.2f}"
          f"  (r2 {fit['r2']:.2f})")
    total_lag = fit["tau_s"] + fit["delay_s"]
    if total_lag > 1e-3:
        print(f"    -> lag-limited bandwidth ~{1.0 / (2 * math.pi * total_lag):.2f} Hz; a cross-track")
        print(f"       correction faster than that will oscillate whatever the gain says")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    for p in sys.argv[1:]:
        report(p.removesuffix(".json").removesuffix("_mppi_diag.csv"))
