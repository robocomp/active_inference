#!/usr/bin/env python3
"""Whiteness test for the wheel-odometry velocity stream at zero command.

WHAT IT DECIDES
    The preintegrator applies a zero-velocity update (ZUPT) once per odometry SAMPLE. That earns
    sqrt(N) of extra confidence if and only if the residual velocity noise is white at the sampling
    interval. If consecutive samples are correlated, the same fact is being counted many times and
    the parked motion prior is too tight -- by up to the full sqrt(5) that the 100 -> 20 ms publish
    period change introduced.

PRE-REGISTERED DECISION RULE (fixed 2026-08-25, before any data was collected)
    WHITE          |rho_1| < 0.1   -> the per-sample ZUPT is legitimate, change nothing
    CORRELATED     rho_1  > 0.3    -> double counting; cure is the density form R = sigma_rest^2/dt
    INCONCLUSIVE   otherwise       -> say so; do not pick a side

INPUT   etc/odom_samples.csv, written by room_concept with RoomConcept.OdomSampleLog = true.
        One line per ARRIVING sample. Do NOT substitute tmp/sdf_localizer/log_*.csv: that is one row
        per lidar sweep carrying only the latest odometry sample, so it aliases this stream onto a
        slower one and would answer a different question.
"""
import sys, csv, math, statistics as st

SETTLE_MS = 1000     # ignore samples within this long after the last non-zero command (deceleration)
CH = ("adv", "side", "rot")


def load(path):
    with open(path, newline="") as f:
        head = f.readline()
        if ";" in head:                      # a comma-decimal file would separate on ';'
            sys.exit("file looks like it was written under a comma-decimal locale -- check the writer")
        f.seek(0)
        return list(csv.DictReader(f))


def autocorr(x, lag):
    n = len(x)
    if n <= lag + 2:
        return float("nan")
    m = sum(x) / n
    d = [v - m for v in x]
    den = sum(v * v for v in d)
    if den <= 0:
        return float("nan")
    return sum(d[i] * d[i + lag] for i in range(n - lag)) / den


def main(path="etc/odom_samples.csv"):
    rows = load(path)
    if not rows:
        sys.exit(f"{path}: no rows")

    # Sampling interval, measured. Use the producer's own clock where it has one: the integrator
    # brackets against that clock, so it is the interval that actually enters the covariance.
    sim = rows[0]["simulated"] == "1" and int(rows[0]["sim_ts_ms"]) > 0
    tkey = "sim_ts_ms" if sim else "source_ts_ms"
    ts = [int(r[tkey]) for r in rows]
    gaps = [b - a for a, b in zip(ts, ts[1:]) if b > a]
    dt_ms = st.median(gaps) if gaps else float("nan")
    dropped = sum(1 for g in gaps if g > 1.5 * dt_ms)

    # Zero command, with a settling margin. A sample taken while the robot is still coasting to a
    # stop is not a sample of a parked robot, and including them inflates sigma with real motion.
    last_moving = None
    keep = []
    for r, t in zip(rows, ts):
        moving = any(abs(float(r[f"cmd_{c}"])) > 0.0 for c in CH)
        if moving:
            last_moving = t
            continue
        if last_moving is not None and t - last_moving < SETTLE_MS:
            continue
        keep.append(r)

    print(f"file            {path}")
    print(f"rows            {len(rows)} total, {len(keep)} at zero command (+{SETTLE_MS} ms settle)")
    print(f"clock           {'SIM (producer)' if sim else 'wall'}")
    print(f"sample interval {dt_ms:.1f} ms  ->  {1000.0 / dt_ms:.1f} Hz" if dt_ms == dt_ms else "")
    print(f"dropped samples {dropped} gaps > 1.5x interval")
    if len(keep) < 200:
        print("\nNOT ENOUGH DATA. Park the robot with the controller idle and collect >= 200 "
              "zero-command samples (>= 4 s at 50 Hz; more is better -- rho_1's own standard error "
              "is about 1/sqrt(N), so 200 samples resolves it only to +/-0.07).")
        return

    print(f"\n{'channel':>8} {'sigma':>12} {'rho_1':>8} {'rho_2':>8} {'rho_3':>8} {'N_eff':>8}  verdict")
    verdicts = {}
    for c in CH:
        x = [float(r[c]) for r in keep]
        s = st.pstdev(x)
        r1, r2, r3 = autocorr(x, 1), autocorr(x, 2), autocorr(x, 3)
        n_eff = len(x) * (1 - r1) / (1 + r1) if -1 < r1 < 1 else float("nan")
        if s == 0.0:
            v = "IDENTICALLY ZERO"
        elif abs(r1) < 0.1:
            v = "WHITE"
        elif r1 > 0.3:
            v = "CORRELATED"
        else:
            v = "inconclusive"
        verdicts[c] = v
        unit = "rad/s" if c == "rot" else "m/s"
        print(f"{c:>8} {s:9.5f} {unit:>5} {r1:8.3f} {r2:8.3f} {r3:8.3f} {n_eff:8.0f}  {v}")

    # What the numbers imply for the ZUPT constants, stated whichever way the verdict falls.
    print()
    for c, key in (("adv", "zupt_density_v"), ("rot", "zupt_density_omega")):
        x = [float(r[c]) for r in keep]
        s = st.pstdev(x)
        if s > 0 and dt_ms == dt_ms:
            dens = s * math.sqrt(dt_ms * 1e-3)
            print(f"{key:18} measured here = {s:.4f} (per-sample), "
                  f"density equivalent = {dens:.5f} /sqrt(s)")
    print("\nThe NoiseModel takes DENSITIES (m/sqrt(s), rad/sqrt(s)); the 'density equivalent'\n"
          "column above is the number to put in PreintZuptDensityV/Omega. P3Bot measured\n"
          "0.01496 / 0.03001 on 2026-08-26; Shadow's old per-sample 0.0202 / 0.0323 were 5.9x and\n"
          "7.3x too tight for this base. Re-measure on any new platform.")
    if all(v == "WHITE" for v in verdicts.values()):
        print("\nVERDICT: white -> the per-sample ZUPT is legitimate; the 50 Hz tightening is earned.")
    elif any(v == "CORRELATED" for v in verdicts.values()):
        print("\nVERDICT: correlated -> the ZUPT double-counts. Cure: R = sigma_rest^2/dt "
              "(density form), which makes the parked prior rate-invariant with no new constant.")
    else:
        print("\nVERDICT: inconclusive by the pre-registered rule. Collect more, or state it as open.")


if __name__ == "__main__":
    main(*sys.argv[1:])
