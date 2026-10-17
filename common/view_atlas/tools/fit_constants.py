#!/usr/bin/env python3
"""Fit the view-atlas constants (VIEW_ATLAS.md §3.5) from a views dataset (build_dataset.py).

For every query view q of an object and every other bin of that SAME object (views >= 10 s apart from q),
build the bin prototype and record (gap D, s_shape, s_colour). Fit
    mu_d(D) = mu_bg_d + (mu0_d - mu_bg_d) exp(-D^2 / 2 ell^2)
and take (mu_own, cov_own) from the pairs with D < 15 deg. The background (mu_bg, cov_bg) comes from the
same computation against OTHER objects of the SAME class at the same object-frame bearing.
repeat_ell_rad is DERIVED, not fitted: the angular step at which the expected shape similarity falls by one
per-view sd, i.e. views closer than that cannot be told from a repeat look. repeat_ell_logr likewise from the
similarity drop with |log range ratio| at near-equal bearing. cov_n = cov_own (per-view noise dominates).
    python3 fit_constants.py <dataset_dir> <out.txt>
    python3 fit_constants.py --selftest
"""
import argparse, datetime, math, os, sys
import numpy as np, pandas as pd
from scipy.optimize import curve_fit

NB = 12; STEP = 2 * math.pi / NB
wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi

def relax(D, mu0, mubg, ell): return mubg + (mu0 - mubg) * np.exp(-D ** 2 / (2 * ell ** 2))

def load(d):
    v = pd.read_csv(os.path.join(d, "views.csv"))
    n = len(v)
    S = np.fromfile(os.path.join(d, "shape.f32"), "<f4").reshape(n, 384)
    C = np.fromfile(os.path.join(d, "colour.f32"), "<f4").reshape(n, 64)
    v["bearing"] = [wrap(math.atan2(r.cam_y - r.obj_y, r.cam_x - r.obj_x) - r.obj_yaw) for r in v.itertuples()]
    return v, S, C

def bin_protos(v, S, C, idx, exclude_t):
    """Per bin of one object's views (excluding views within 10 s of exclude_t): (mean bearing, shape proto)."""
    out = {}
    keep = idx[np.abs(v.stamp_ms.values[idx] / 1000 - exclude_t) >= 10]
    b = ((v.bearing.values[keep] + math.pi) // STEP).astype(int) % NB
    for k in np.unique(b):
        m = keep[b == k]; s = S[m].sum(0)
        out[k] = (math.atan2(np.sin(v.bearing.values[m]).sum(), np.cos(v.bearing.values[m]).sum()), s / np.linalg.norm(s),
                  C[m].sum(0) / np.linalg.norm(C[m].sum(0)))
    return out

def pairs(v, S, C, rng, max_q=4000):
    own, bg = [], []
    by_inst = {i: np.where(v.inst.values == i)[0] for i in v.inst.unique()}
    qs = rng.permutation(len(v))[:max_q]
    for q in qs:
        i, c = v.inst.values[q], v.cls.values[q]; tq = v.stamp_ms.values[q] / 1000
        for k, (bb, ps, pc) in bin_protos(v, S, C, by_inst[i], tq).items():
            own.append((abs(wrap(v.bearing.values[q] - bb)), S[q] @ ps, C[q] @ pc))
        for j, idx in by_inst.items():
            if j == i or v.cls.values[idx[0]] != c: continue
            for k, (bb, ps, pc) in bin_protos(v, S, C, idx, -1e9).items():
                D = abs(wrap(v.bearing.values[q] - bb))
                if D < math.radians(15): bg.append((D, S[q] @ ps, C[q] @ pc))
    return np.array(own), np.array(bg)

def fit(own, bg, logr_pairs=None):
    p, _ = curve_fit(relax, own[:, 0], own[:, 1], p0=[0.95, 0.85, 0.8], bounds=([0, -1, 0.05], [1, 1, math.pi]))
    mu0, mubg_d_curve, ell = p
    near = own[own[:, 0] < math.radians(15)]
    mu_own = near[:, 1:].mean(0); cov_own = np.cov(near[:, 1:].T)
    mu_bg = bg[:, 1:].mean(0); cov_bg = np.cov(bg[:, 1:].T)
    sd = math.sqrt(cov_own[0, 0]); drop = max(mu0 - mubg_d_curve, 1e-6)
    ratio = sd / drop
    if ratio >= 0.5:
        print(f"WARNING: repeat derivation ill-conditioned: sd/drop = {ratio:.3f}" +
              ("; >= 1: undefined, repeat_ell_rad is a clamp artefact" if ratio >= 1 else ""))
    repeat = ell * math.sqrt(-2 * math.log(max(1 - sd / drop, 1e-6)))
    # small-step model s(x) = mu0 - a x^2 in x = |log range ratio|; a repeat is a step whose drop a x^2 < sd
    if logr_pairs is None or len(logr_pairs) < 20:
        repeat_logr = 0.2                                     # too few same-bearing pairs: keep the default
    else:
        a = max(np.mean(mu0 - logr_pairs[:, 1]), 1e-6) / max(np.mean(logr_pairs[:, 0] ** 2), 1e-6)
        repeat_logr = float(math.sqrt(sd / a))
    return dict(_repeat_ratio=ratio, ell_rad=ell, repeat_ell_rad=repeat, repeat_ell_logr=repeat_logr,
                mu_own_d=mu_own[0], mu_own_c=mu_own[1], mu_bg_d=mu_bg[0], mu_bg_c=mu_bg[1],
                cov_own_dd=cov_own[0, 0], cov_own_dc=cov_own[0, 1], cov_own_cc=cov_own[1, 1],
                cov_bg_dd=cov_bg[0, 0], cov_bg_dc=cov_bg[0, 1], cov_bg_cc=cov_bg[1, 1],
                cov_n_dd=cov_own[0, 0], cov_n_dc=cov_own[0, 1], cov_n_cc=cov_own[1, 1])

def logr_pairs(v, S):
    """Same object, near-equal bearing (< 5 deg), >= 10 s apart (as the own pairs), different range:
    (|log r ratio|, shape similarity). Up to 5 random matches per anchor (seeded)."""
    out = []; rng = np.random.default_rng(0)
    t = v.stamp_ms.values / 1000
    for i in v.inst.unique():
        idx = np.where(v.inst.values == i)[0]
        for a_ in idx[:300]:
            d = np.abs([wrap(v.bearing.values[a_] - v.bearing.values[b]) for b in idx])
            m = idx[(d < math.radians(5)) & (idx != a_) & (np.abs(t[idx] - t[a_]) >= 10)]
            for b in rng.permutation(m)[:5]:
                out.append((abs(math.log(v.range_m.values[a_] / v.range_m.values[b])), S[a_] @ S[b]))
    return np.array(out)

def selftest():
    rng = np.random.default_rng(0); truth = dict(mu0=0.95, mubg=0.86, ell=0.85)
    D = rng.uniform(0, math.pi, 4000)
    sd = relax(D, truth["mu0"], truth["mubg"], truth["ell"]) + rng.normal(0, 0.02, D.size)
    sc = 0.97 + rng.normal(0, 0.02, D.size)
    own = np.c_[D, sd, sc]; bg = np.c_[rng.uniform(0, 0.25, 2000), 0.86 + rng.normal(0, 0.04, 2000), 0.5 + rng.normal(0, 0.2, 2000)]
    k = fit(own, bg)
    ok = abs(k["ell_rad"] - truth["ell"]) / truth["ell"] < 0.1 and abs(k["mu_bg_d"] - 0.86) < 0.01 and abs(k["mu_own_c"] - 0.97) < 0.01
    print(f"selftest: ell {k['ell_rad']:.3f} (truth 0.85), mu_bg_d {k['mu_bg_d']:.3f}, mu_own_c {k['mu_own_c']:.3f} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dataset", nargs="?"); ap.add_argument("out", nargs="?")
    ap.add_argument("--selftest", action="store_true"); a = ap.parse_args()
    if a.selftest: sys.exit(selftest())
    v, S, C = load(a.dataset); rng = np.random.default_rng(0)
    own, bg = pairs(v, S, C, rng)
    k = fit(own, bg, logr_pairs(v, S))
    ratio = k.pop("_repeat_ratio")
    with open(a.out, "w") as f:
        f.write(f"# view_atlas constants, fitted {datetime.date.today()} by fit_constants.py\n"
                f"# dataset: {os.path.abspath(a.dataset)} ({len(v)} views, {v.inst.nunique()} objects)\n"
                f"# pairs: own {len(own)}, background {len(bg)}\n"
                f"# repeat derivation: sd/drop = {ratio:.6g}\n")
        for key, val in k.items(): f.write(f"{key} = {val:.9g}\n")
    print(open(a.out).read())

if __name__ == "__main__":
    main()
