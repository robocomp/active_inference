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
            own.append((abs(wrap(v.bearing.values[q] - bb)), S[q] @ ps, C[q] @ pc, v.mask_px.values[q]))
        for j, idx in by_inst.items():
            if j == i or v.cls.values[idx[0]] != c: continue
            for k, (bb, ps, pc) in bin_protos(v, S, C, idx, -1e9).items():
                D = abs(wrap(v.bearing.values[q] - bb))
                if D < math.radians(15): bg.append((D, S[q] @ ps, C[q] @ pc))
    return np.array(own), np.array(bg)

def fit_noise(near, n_bands=6):
    """near: own pairs at gap < 15 deg, columns (D, s_d, s_c, px). Returns (cov_own, cov_n, px_scale, sse, n_bands)."""
    edges = np.quantile(near[:, 3], np.linspace(0, 1, n_bands + 1))
    bands = [near[(near[:, 3] >= lo) & (near[:, 3] <= hi)] for lo, hi in zip(edges[:-1], edges[1:])]
    bands = [b for b in bands if len(b) >= 30]
    px = np.array([np.median(b[:, 3]) for b in bands])
    C = np.array([np.cov(b[:, 1:3].T) for b in bands])            # (B, 2, 2)
    best = None
    for scale in np.geomspace(100, 50000, 64):
        q = 1 - np.exp(-px / scale); x = (1 - q) / q                 # C(b) = A + x_b * N, elementwise
        X = np.c_[np.ones_like(x), x]
        coef, res, *_ = np.linalg.lstsq(X, C.reshape(len(px), 4), rcond=None)
        sse = float(((X @ coef - C.reshape(len(px), 4)) ** 2).sum())
        if best is None or sse < best[3]:
            best = (coef[0].reshape(2, 2), coef[1].reshape(2, 2), float(scale), sse)
    return best + (len(bands),)

def fit(own, bg, logr_pairs=None):
    p, _ = curve_fit(relax, own[:, 0], own[:, 1], p0=[0.95, 0.85, 0.8], bounds=([0, -1, 0.05], [1, 1, math.pi]))
    mu0, mubg_d_curve, ell = p
    near = own[own[:, 0] < math.radians(15)]
    mu_own = near[:, 1:3].mean(0)
    cov_own_fit, cov_n_fit, px_scale, sse, nb = fit_noise(near)
    for nm, M in (("cov_own", cov_own_fit), ("cov_n", cov_n_fit)):
        ev = np.linalg.eigvalsh(M)
        if ev.min() <= 0: print(f"WARNING: fitted {nm} not positive definite: {ev}")
    # Two roles of "cov_n" (controller ruling): fit_noise measures the EXCESS covariance of small masks. Branch on
    # whether that excess is positive definite.
    #  * PD excess: the data support a mask-size term; write cov_n = excess and quality_px_scale = best scale.
    #  * not PD: the excess is not supported (here: covariance is flat in mask size). cov_n then keeps its phase-1
    #    meaning, the per-view scatter used by the prototype-uncertainty term: cov_own = cov_n = the POOLED
    #    np.cov of the near pairs (not the model intercept, which extrapolates an unsupported model), also used
    #    for the repeat-scale sd. quality_px_scale is NOT written (stays the flagged AtlasParams default). The finding goes in the header.
    ev_n = np.linalg.eigvalsh(cov_n_fit)
    excess_ok = bool(ev_n.min() > 0)
    px_lo, px_hi = float(near[:, 3].min()), float(near[:, 3].max())
    edge = bool(px_scale >= 50000 * (1 - 1e-9) or px_scale <= 100 * (1 + 1e-9))
    if excess_ok:
        noise_header = f"noise model: cov(px) = cov_own + cov_n (1-q)/q, q = 1-exp(-px/quality_px_scale); bands={nb}, sse={sse:.3g}"
        cov_n_out, scale_out = cov_n_fit, {"quality_px_scale": px_scale}
        cov_own_out = cov_own_fit
    else:
        noise_header = (f"noise vs mask size: excess term NOT supported by the data over px [{px_lo:.0f}, {px_hi:.0f}] "
                        f"(fitted excess cov eigenvalues {ev_n[0]:.3g}, {ev_n[1]:.3g}; best scale {px_scale:.0f} at grid edge = "
                        f"{'yes' if edge else 'no'}; bands={nb}, sse={sse:.3g}); cov_n = per-view scatter, "
                        f"quality_px_scale left at its flagged default")
        cov_own_out = cov_n_out = np.cov(near[:, 1:3].T); scale_out = {}
    mu_bg = bg[:, 1:3].mean(0); cov_bg = np.cov(bg[:, 1:3].T)
    sd = math.sqrt(cov_own_out[0, 0]); drop = max(mu0 - mubg_d_curve, 1e-6)
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
    return dict(_repeat_ratio=ratio, _noise_header=noise_header, ell_rad=ell, repeat_ell_rad=repeat, repeat_ell_logr=repeat_logr,
                mu_own_d=mu_own[0], mu_own_c=mu_own[1], mu_bg_d=mu_bg[0], mu_bg_c=mu_bg[1],
                cov_own_dd=cov_own_out[0, 0], cov_own_dc=cov_own_out[0, 1], cov_own_cc=cov_own_out[1, 1],
                cov_bg_dd=cov_bg[0, 0], cov_bg_dc=cov_bg[0, 1], cov_bg_cc=cov_bg[1, 1],
                cov_n_dd=cov_n_out[0, 0], cov_n_dc=cov_n_out[0, 1], cov_n_cc=cov_n_out[1, 1], **scale_out)

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
    own = np.c_[D, sd, sc, np.full(D.size, 5000.0)]; bg = np.c_[rng.uniform(0, 0.25, 2000), 0.86 + rng.normal(0, 0.04, 2000), 0.5 + rng.normal(0, 0.2, 2000)]
    k = fit(own, bg)
    ok = abs(k["ell_rad"] - truth["ell"]) / truth["ell"] < 0.1 and abs(k["mu_bg_d"] - 0.86) < 0.01 and abs(k["mu_own_c"] - 0.97) < 0.01
    print(f"selftest: ell {k['ell_rad']:.3f} (truth 0.85), mu_bg_d {k['mu_bg_d']:.3f}, mu_own_c {k['mu_own_c']:.3f} -> {'PASS' if ok else 'FAIL'}")
    pxs = np.exp(rng.uniform(np.log(300), np.log(30000), 6000)); qq = 1 - np.exp(-pxs / 2000.0)
    var = 4e-4 + 2e-3 * (1 - qq) / qq
    syn = np.c_[np.zeros(6000), 0.95 + rng.normal(0, 1, 6000) * np.sqrt(var), 0.97 + rng.normal(0, 1, 6000) * np.sqrt(var), pxs]
    a_, n_, s_, _, _ = fit_noise(syn)
    ok2 = (2000 / 1.5 < s_ < 2000 * 1.5) and abs(n_[0, 0] - 2e-3) / 2e-3 < 0.3 and abs(n_[1, 1] - 2e-3) / 2e-3 < 0.3
    print(f"selftest noise: scale {s_:.0f} (truth 2000), cov_n_dd {n_[0,0]:.2e} (truth 2.0e-03) -> {'PASS' if ok2 else 'FAIL'}")
    ok = ok and ok2
    # flat in px: pure sampling noise, so the excess is PD only by chance; fixed seed keeps the case deterministic
    r2 = np.random.default_rng(0); Df = r2.uniform(0, math.pi, 6000)
    flat = np.c_[Df, relax(Df, 0.95, 0.86, 0.85) + r2.normal(0, 0.04, 6000), 0.97 + r2.normal(0, 0.08, 6000),
                 np.exp(r2.uniform(np.log(300), np.log(30000), 6000))]
    kf = fit(flat, bg)
    ok3 = "quality_px_scale" not in kf and all(kf[f"cov_n_{c}"] == kf[f"cov_own_{c}"] for c in ("dd", "dc", "cc"))
    print(f"selftest flat-in-px through fit(): no quality_px_scale, cov_n == cov_own -> {'PASS' if ok3 else 'FAIL'}")
    ok = ok and ok3
    return 0 if ok else 1

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dataset", nargs="?"); ap.add_argument("out", nargs="?")
    ap.add_argument("--selftest", action="store_true"); a = ap.parse_args()
    if a.selftest: sys.exit(selftest())
    v, S, C = load(a.dataset); rng = np.random.default_rng(0)
    own, bg = pairs(v, S, C, rng)
    k = fit(own, bg, logr_pairs(v, S))
    ratio = k.pop("_repeat_ratio"); nh = k.pop("_noise_header")
    try: stride = [l.split("stride=", 1)[1].split()[0] for l in open(os.path.join(a.dataset, "dataset_meta.txt")) if "stride=" in l][0]
    except Exception: stride = "unknown"
    with open(a.out, "w") as f:
        f.write(f"# view_atlas constants, fitted {datetime.date.today()} by fit_constants.py\n"
                f"# dataset: {os.path.abspath(a.dataset)} ({len(v)} views, {v.inst.nunique()} objects)\n"
                f"# pairs: own {len(own)}, background {len(bg)}\n"
                f"# dataset: stride={stride}\n"
                f"# {nh}\n"
                f"# repeat derivation: sd/drop = {ratio:.6g}\n")
        for key, val in k.items(): f.write(f"{key} = {val:.9g}\n")
    print(open(a.out).read())

if __name__ == "__main__":
    main()
