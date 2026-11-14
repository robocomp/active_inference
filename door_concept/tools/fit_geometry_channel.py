#!/usr/bin/env -S python3 -I
"""Fit the geometry channel's mu1 spread (`geometry,mu1_sd` of etc/door_detectors.csv; core/existence_update.h geo_mu1_sd, fixed at 2 until now).

The channel reads the leaf contour's edge support x against the controls' mean mu0 and spread s (existence_update.cpp: x = s_true/ref,
mu0 = mean of the controls, s = their std). Its model: a frame of NO door reads x ~ N(mu0, s^2); a frame of a door reads
x ~ N(mu1, s^2) with mu1 = mu0 + k * s. The standardised z = (x - mu0) / s is therefore N(0, 1) without a door and N(k, 1) with one, and k
(`mu1_sd`) is the location of z over the frames of TRUE doors. The log repeats a frame's value on every cycle until the next frame, so
the frames are first reduced to DISTINCT (door, geo_x, geo_mu0, geo_s) tuples (one per real frame). The location is the MEDIAN (maximum
likelihood under a Laplace error), not the mean (ML under the Gaussian the model assumes): on the three apartment tapes the label-1 z has
a heavy tail from frames whose control spread s collapses (0.05 against a median 2.5), where the Gaussian mean read 0.02 / 26 / 4.6 per
tape and is a few outliers, not a location; both are printed. The label-0 frames are not used by the fit; their median / sd of z are printed
as a check of the null (0, 1).

Input columns: geo_x, geo_mu0, geo_s, label (1 = a frame of a true Webots door, 0 = not); tools/label_door_views.py writes
RUN/door_geo_pairs.csv. Fewer than MIN_N label-1 frames: not identified, the compiled value (2.0, n = 0) is kept. The estimate's standard error is the median's
(1.2533 * robust sd / sqrt(n), robust sd = IQR / 1.349, which the collapsed-s tail does not inflate). When the estimate is within 2 standard
errors of 0 (or negative: the loader's domain is mu1_sd >= 0) the row is written as 0 = "no information" (mu1 = mu0, the geometry
log-likelihood-ratio is exactly 0) with the real n and the source tagged "non-informative".
Usage: python3 -I tools/fit_geometry_channel.py PAIRS.csv [PAIRS.csv ...] [--dry-run] [--out etc/door_detectors.csv] [--source TAG]
Writes the `geometry,mu1_sd` row only without --dry-run; every other row of the output file is carried over.
"""
import argparse, csv, datetime, math, os, sys
import numpy as np

DEFAULT = 2.0
MIN_N = 20   # kept: a mean of fewer than 20 correlated frames is not a measurement (a count of evidence, not a model term)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+"); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="etc/door_detectors.csv"); ap.add_argument("--source", default=None)
    a = ap.parse_args()
    rows = []
    for p in a.pairs:
        for r in csv.DictReader(open(p, newline="")):
            r["_file"] = p
            rows.append(r)
    x = np.array([float(r["geo_x"]) for r in rows]); m0 = np.array([float(r["geo_mu0"]) for r in rows])
    s = np.array([float(r["geo_s"]) for r in rows]); y = np.array([float(r["label"]) for r in rows])
    ok = np.isfinite(x) & np.isfinite(m0) & np.isfinite(s) & (s > 0)   # s > 0: z is undefined for a control set with no spread (the channel adds no term there either)
    z = (x - m0) / np.where(ok, s, 1.0)
    # distinct frames: the first row of each (file, door, geo_x, geo_mu0, geo_s)
    seen, keep = set(), np.zeros(len(rows), bool)
    for i, r in enumerate(rows):
        k = (r["_file"], r["door"], r["geo_x"], r["geo_mu0"], r["geo_s"])
        if k not in seen:
            seen.add(k); keep[i] = True
    n_rep = int((ok & (y == 1)).sum())
    ok &= keep
    z1, z0 = z[ok & (y == 1)], z[ok & (y == 0)]
    print(f"{len(z1)} distinct label-1 frames (of {n_rep} logged rows), {len(z0)} distinct label-0 frames")
    if len(z0):
        print(f"label 0: mean z {z0.mean():+.3f} sd {z0.std():.3f}  (null model: 0, 1)")
    noninf = False
    if len(z1) < MIN_N:
        print(f"mu1_sd: {len(z1)} label-1 frames, not fitted (compiled value {DEFAULT} kept)")
        k, n = DEFAULT, 0
    else:
        k, n = float(np.median(z1)), len(z1)
        q = np.percentile(z1, [10, 25, 75, 90])
        iqr = float(np.percentile(z1, 75) - np.percentile(z1, 25))
        se = 1.2533 * (iqr / 1.349) / math.sqrt(n)
        print(f"standard error of the median {se:.4f}")
        if k < 2 * se:   # kept: a decision about whether the data says anything at all, made once by the fitter, not a term of the likelihood
            print(f"estimate {k:+.4f} is within 2 standard errors of 0 (or negative): written as 0, non-informative")
            k = 0.0; noninf = True
        print(f"mu1_sd={k:.4f}  (n={n} distinct frames; Gaussian mean {z1.mean():.3f}, sd {z1.std():.3f}; z quantiles 10/25/75/90 % "
              f"{q[0]:.2f} {q[1]:.2f} {q[2]:.2f} {q[3]:.2f})")
    if a.dry_run:
        return 0
    src = (a.source or f"fit_geometry_channel.py {datetime.date.today()}") if n > 0 else "compiled_defaults"
    if noninf: src += " non-informative"
    kept = [ln for ln in open(a.out) if not ln.startswith("geometry,")] if os.path.exists(a.out) else ["camera,param,value,n,source\n"]
    with open(a.out, "w", newline="") as o:
        o.writelines(kept)
        o.write(f"geometry,mu1_sd,{k:.6g},{n},{src}\n")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
