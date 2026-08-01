#!/usr/bin/env python3
"""mppi_diag.py — is the MPPI optimising, or averaging?

Reads mppi_diag.csv (one row per control cycle) and answers three questions that behaviour cannot:

  1. EFFECTIVE SAMPLE SIZE. ESS = (sum w)^2 / sum w^2 over the softmax weights, between 1 and K.
     At ESS ~ K every rollout carries the same weight, the command is the plain mean of the samples,
     and the costs have had no influence at all — a robot in that state creeps regardless of how the
     terms are weighted, and no amount of cost tuning will move it.
  2. WHICH TEMPERATURE IS IN FORCE. lambda_used = max(lambda_adaptive, cost_range/5). When the second
     argument wins, the temperature is a function of the cost SPREAD rather than a tuned quantity, and
     the weights are prevented from ever becoming peaked. That is a mechanism, not a setting.
  3. WHICH TERM OWNS THE COST. If one term dominates the total everywhere, the others are decoration:
     they cannot change the ranking of the rollouts no matter what their weights say.

  scripts/mppi_diag.py [mppi_diag.csv]
"""
import csv
import sys
from statistics import mean, median


def pct(xs, p):
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(len(s) - 1, int(p * (len(s) - 1)))]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "mppi_diag.csv"
    try:
        rows = [r for r in csv.DictReader(open(path))]
    except OSError:
        print(f"cannot open {path}")
        return 1
    rows = [r for r in rows if r.get("ess")]
    if not rows:
        print("no rows")
        return 1

    def col(name):
        out = []
        for r in rows:
            try:
                out.append(float(r[name]))
            except (KeyError, ValueError, TypeError):
                pass        # a partially-written last row: this file is often read while a lap is running
        return out

    ess, K = col("ess"), col("ess_K")
    ratio = col("ess_ratio")
    lu, la = col("lambda_used"), col("lambda_adaptive")
    rng, best = col("cost_range"), col("cost_best")
    adv, meas = col("cmd_adv"), col("meas_speed")
    print(f"{len(rows)} cycles, K = {int(median(K)) if K else '?'}")

    # 1 ── ESS
    print("\n1. EFFECTIVE SAMPLE SIZE (fraction of K)")
    print(f"   mean {mean(ratio):.3f}   p05 {pct(ratio,0.05):.3f}   p50 {pct(ratio,0.50):.3f}   "
          f"p95 {pct(ratio,0.95):.3f}")
    flat = sum(1 for x in ratio if x > 0.5) / len(ratio)
    peaked = sum(1 for x in ratio if x < 0.1) / len(ratio)
    print(f"   {100*flat:5.1f}% of cycles above 0.50 of K  (averaging: costs barely rank the rollouts)")
    print(f"   {100*peaked:5.1f}% of cycles below 0.10 of K  (decisive)")
    print(f"   mean ESS {mean(ess):.1f} of {int(median(K))} samples")

    # 2 ── which temperature
    print("\n2. TEMPERATURE")
    floor_binds = [1 if u > a + 1e-6 else 0 for u, a in zip(lu, la)]
    print(f"   lambda_used      mean {mean(lu):8.2f}   p50 {pct(lu,0.5):8.2f}   p95 {pct(lu,0.95):8.2f}")
    print(f"   lambda_adaptive  mean {mean(la):8.2f}   p50 {pct(la,0.5):8.2f}")
    print(f"   cost_range       mean {mean(rng):8.2f}   p50 {pct(rng,0.5):8.2f}")
    if floor_binds:
        print(f"   the cost-range FLOOR sets the temperature in {100*mean(floor_binds):.1f}% of cycles")
        if mean(floor_binds) > 0.5:
            print("   -> the temperature is being driven by the spread of the costs, so the weights cannot")
            print("      become peaked no matter what lambda is configured to.")

    # 3 ── who owns the cost
    print("\n3. COST COMPOSITION (best rollout, mean over cycles)")
    terms = ["g_goal", "g_obs", "g_vel", "g_smooth", "g_lat", "g_cbf"]
    means = {t: (mean(col(t)) if col(t) else 0.0) for t in terms}
    tot = sum(means.values()) or 1.0
    for t, v in sorted(means.items(), key=lambda kv: -kv[1]):
        bar = "#" * int(round(40 * v / tot))
        print(f"   {t:9} {v:8.3f}  {100*v/tot:5.1f}%  {bar}")
    print(f"   total {tot:.3f}   (cost_best mean {mean(best):.3f})")

    # 4 ── speed, for context
    print("\n4. SPEED")
    print(f"   commanded mean {mean(adv):.3f} m/s   p95 {pct(adv,0.95):.3f}")
    print(f"   measured  mean {mean(meas):.3f} m/s   p95 {pct(meas,0.95):.3f}")
    esdf = col("min_esdf")
    if esdf:
        print(f"   min_esdf over the horizon: mean {mean(esdf):.3f} m   p05 {pct(esdf,0.05):.3f} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
