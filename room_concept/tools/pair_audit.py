#!/usr/bin/env python3
"""Stage 1 of VALIDATION_THREE_DEVICE_CORNERS: audit the camera<->LiDAR correspondence with the
extrinsics FROZEN, on data already recorded.

★ WHAT THIS TOOL WILL AND WILL NOT TELL YOU.
  The winning association's cost (`assoc_chi2`) is TRUNCATED at CornerDetector::Params::assoc_chi2
  = 5.991 by the gate itself: an over-gate pair is left INFEASIBLE and never assigned, so no emitted
  row can exceed the bound. Its distribution therefore carries NO information about how often the
  gate was right, and any "x% exceed the 95% bound" computed from it is a statement about the gate's
  arithmetic, not about the data. The quantities that CAN judge the association are `n_rivals` and
  `runnerup_chi2` -- what the gate had to choose between -- and those exist only in files written by
  a binary built after 2026-09-02. Older files are audited on residual structure alone, and this
  tool says so rather than printing a margin column of NaNs.

Locale: python's float() is locale-independent, so CLAUDE.md's from_chars rule does not bind here.
Usage:  pair_audit.py etc/image_edge_pair_ricoh.csv [more...]
"""
import sys, numpy as np

GATE = 5.991           # CornerDetector::Params::assoc_chi2, compiled in
DEG_PER_PX = {"zed": 0.128, "ricoh": 0.188}   # a pixel is a different angle on each camera

def load(path):
    d = np.genfromtxt(path, delimiter=',', names=True, invalid_raise=False,
                      dtype=None, encoding='utf-8')
    return d

def audit(path):
    d = load(path)
    cols = set(d.dtype.names)
    n = d.shape[0]
    print(f"\n{'='*78}\n{path}   {n} rows")
    cam = "not recorded (pre-2026-09-02 log)"
    if 'camera' in cols:
        u = sorted(set(str(x) for x in d['camera']))
        cam = ",".join(u) + ("  ⚠ MIXTURE" if len(u) > 1 else "")
    print(f"camera: {cam}")

    r = np.hypot(d['ru'], d['rv'])
    s = np.hypot(d['sigu'], d['sigv'])
    ok = np.isfinite(r) & np.isfinite(s) & (s > 0)
    r, s = r[ok], s[ok]
    v = d['vertex'][ok].astype(int)
    c = d['assoc_chi2'][ok]
    p = d['assoc_prob'][ok]

    print(f"\n-- residual vs its own sigma (the pair's, not the association's) --")
    print(f"   |r|  med {np.median(r):7.2f} px   p99 {np.percentile(r,99):8.1f}   max {r.max():9.1f}")
    print(f"   sig  med {np.median(s):7.2f} px   p99 {np.percentile(s,99):8.1f}   max {s.max():9.1f}")
    nr = r / s
    # 2-DoF: P(|r|/sigma > 2.448) = 5% when the covariance is correctly scaled.
    print(f"   |r|/sig med {np.median(nr):.3f}   frac>2.448 {100*(nr>2.448).mean():5.2f}%  (5% if calibrated)")
    print(f"   -> covariance is {'TIGHT (optimistic)' if (nr>2.448).mean()>0.05 else 'CONSERVATIVE'}"
          f" by roughly {np.median(nr)/0.8326:.2f}x on the median row")

    print(f"\n-- the association gate, and why its cost cannot judge it --")
    print(f"   assoc_chi2 max {c.max():.4f} against a gate of {GATE}  ->"
          f" {(c>GATE).sum()} rows above it, and there can never be any")
    print(f"   assoc_chi2 quantiles 50/90/99/99.9: "
          + "/".join(f"{q:.2f}" for q in np.percentile(c,[50,90,99,99.9])))
    print(f"   assoc_prob == 1.0 on {100*(p>=0.999999).mean():.1f}%  (< 0.5 on {100*(p<0.5).mean():.2f}%)")
    if 'n_rivals' in cols and 'runnerup_chi2' in cols:
        nrv = d['n_rivals'][ok].astype(int)
        ru  = d['runnerup_chi2'][ok]
        has = nrv > 0
        print(f"   n_rivals: 0 on {100*(~has).mean():.1f}% of rows -- those had NO choice to get wrong")
        if has.any():
            margin = ru[has] / np.maximum(c[has], 1e-6)
            print(f"   MARGIN runnerup/accepted, over the {has.sum()} contested rows:")
            print(f"     quantiles 1/10/50/90: "
                  + "/".join(f"{q:.2f}" for q in np.percentile(margin,[1,10,50,90])))
            print(f"     margin < 2 (near coin flip) on {100*(margin<2).mean():.1f}% of contested rows")
    else:
        print("   ⚠ n_rivals / runnerup_chi2 ABSENT -- this file predates the margin logging, so the")
        print("     correspondence CANNOT be audited from it. Re-record before trusting any verdict.")

    print(f"\n-- per-vertex residual structure (frozen extrinsics; mount is ~1 px, so a large")
    print(f"   per-vertex MEAN is a per-corner offset or a mis-association, not the mount) --")
    print(f"   {'vtx':>4} {'n':>7} {'mean_ru':>8} {'mean_rv':>8} {'sd_ru':>8} {'sd_rv':>8} {'med|r|':>7}")
    rows = []
    for vv in sorted(set(v.tolist())):
        m = v == vv
        if m.sum() < 200: continue
        ru_, rv_ = d['ru'][ok][m], d['rv'][ok][m]
        rows.append((vv, m.sum(), ru_.mean(), rv_.mean(), ru_.std(), rv_.std(), np.median(r[m])))
    rows.sort(key=lambda t: -abs(t[2]))
    for t in rows[:12]:
        print(f"   {t[0]:>4} {t[1]:>7} {t[2]:>8.2f} {t[3]:>8.2f} {t[4]:>8.2f} {t[5]:>8.2f} {t[6]:>7.2f}")
    if len(rows) > 12: print(f"   ... {len(rows)-12} more vertices")

    # ── THE ONE THAT DECIDES WHETHER THE MOUNT SOLVE MEANS ANYTHING ──────────────────────────────
    # The pooled estimator adds one independent measurement per ROW. The rows are not independent:
    # they are ~20 vertex clusters, each with its own offset, sampled thousands of times. Ignoring
    # that inflates the effective sample size by n_rows/n_clusters and shrinks the posterior sigma by
    # its square root. Comparing the two is how you find out whether a tight sigma is precision or
    # bookkeeping -- and the mount value is only meaningful against the CLUSTER-ROBUST one.
    deg_px = DEG_PER_PX.get(cam.split(',')[0], None)
    allmean = np.array([t[2] for t in rows]); ns = np.array([t[1] for t in rows], float)
    sd  = allmean.std(ddof=1)
    sem = sd / np.sqrt(len(rows))
    print(f"\n-- clustering: {len(rows)} vertices carry {int(ns.sum())} rows --")
    print(f"   between-vertex spread of mean ru : sd {sd:.2f} px")
    print(f"   sample-weighted mean ru (what the pooled solve sees) : {np.average(allmean,weights=ns):.2f} px")
    print(f"   CLUSTER-ROBUST standard error on that shift          : {sem:.3f} px", end='')
    print(f"  = {sem*deg_px:.4f} deg" if deg_px else "  (deg unknown: camera not recorded)")
    print(f"   design effect sqrt(n_rows/n_clusters) = {np.sqrt(ns.sum()/len(rows)):.0f}x")
    print(f"   -> a per-row posterior sigma is understated by about that factor. A mount estimate")
    print(f"      smaller than the cluster-robust error is not a measurement of the mount.")

for a in sys.argv[1:]: audit(a)
