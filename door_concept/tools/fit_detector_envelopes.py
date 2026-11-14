#!/usr/bin/env -S python3 -I
"""Maximum-likelihood fit of the per-camera detector envelope (rc::detect::DetectorEnvelope: min_fill, max_fill, soft) and hit_rate
from PER-VIEW rows of etc/door_views_log.csv (Task 8: the agent writes one row each time a frame joins a view, carrying that frame's
own q / fill / vis / outcome (fix round 1: the four refer to the same frame); the rows of a (door, camera, view_id) are averaged into
ONE trial per view, its outcome the fraction of its frames with a detection).

Input columns: cycle, door, camera (zed|ricoh), view_id, q, fill_h, fill_v, vis, kappa, hit (0/1 or an associated weight in [0,1]).
Optional: alive_end (1 = the door ended alive with L>0), crossed (1 = the door was crossed). Rows are kept only for doors that ended
alive AND were crossed; a file without those columns is used whole (every door is taken as alive at the end, stated on stdout).
Model, exactly detector_model.h's: P(hit) = env(fill_h) * env(fill_v) * aspect(=1 if absent) * vis * kappa * hit_rate, with
env(f) = sig((f - min_fill)/soft) * sig((max_fill - f)/soft)  (rc::detect::p_detect at visible_frac 1). `q` in the file is ignored.
LABELS (Task 5; tools/label_door_views.py writes them from the Webots world's true doors): an optional column `label` (1 = the view's
door is a true door, 0 = it is not). With it the envelope and hit_rate are fitted on the label-1 views (this replaces the
alive_end/crossed proxy for "a real door") and false_alarm on the label-0 views: a false alarm passes through the same front end as a
door, so P(report | no door) = false_alarm * env(fill_h) * env(fill_v) * aspect * vis * kappa, one scalar fitted by maximum likelihood
(fewer than MIN_FA_VIEWS label-0 views with a non-zero front end: not identifiable, the compiled default is kept with n = 0). Without
`label`, false_alarm is not identifiable from views of real doors and is kept at the compiled default.
--fix-envelope: keep the compiled envelope (min_fill, max_fill, soft) and fit only hit_rate (and false_alarm). Use it when the views are too
few to identify the envelope: on the three apartment tapes there are 43 zed / 59 ricoh views of true doors and the free fit ran its
shoulders to the bounds (zed min_fill 0.02, max_fill 2.2, soft 0.01), the envelope having been fitted already on 7 doors' views
(views_fill_2026-11-12: thousands of frames).
Several logs may be given; their doors are kept apart. A camera with fewer than MIN_VIEWS views keeps its compiled row (n = 0).
Usage: python3 -I tools/fit_detector_envelopes.py LOG.csv [LOG.csv ...] [--dry-run] [--fix-envelope] [--out etc/door_detectors.csv] [--source TAG]
Writes the table only without --dry-run; rows of the output file that are not zed/ricoh (support, confidence, geometry, lidar) are carried over.
"""
import argparse, csv, datetime, math, os, sys
import numpy as np

DEFAULT_FA = {"zed": 0.02, "ricoh": 0.05}
# the compiled defaults (core/detector_llr.h default_detectors()): min_fill, max_fill, soft, hit_rate
DEFAULT_ROW = {"zed": (0.10, 1.50, 0.06, 0.9), "ricoh": (0.216, 1.5, 0.06, 0.5)}
MIN_VIEWS = 20      # kept: fewer views than this and the 4-parameter envelope is not identified (a count of evidence, not a model term)
MIN_FA_VIEWS = 20   # kept: likewise for the one false-alarm scalar
START = {"zed": (0.10, 0.60, 0.06, 0.9), "ricoh": (0.216, 1.5, 0.06, 0.5)}


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def predict(p, fh, fv, asp, vis, kap):
    mn, mx, s, h = p
    e = lambda f: sig((f - mn) / s) * sig((mx - f) / s)
    return e(fh) * e(fv) * asp * vis * kap * h


def nll(p, d):
    mn, mx, s, h = p
    if not (0.0 < h < 1.0 and s > 1e-3 and mx > mn + 0.02 and mn > 0):
        return 1e18
    q = np.clip(predict(p, d["fh"], d["fv"], d["asp"], d["vis"], d["kap"]), 1e-9, 1 - 1e-9)
    return -float(np.sum(d["y"] * np.log(q) + (1 - d["y"]) * np.log1p(-q)))


def nelder_mead(f, x0, iters=1500):
    n = len(x0)
    sim = [np.array(x0, float)]
    for i in range(n):
        x = np.array(x0, float); x[i] *= 1.15; sim.append(x)
    fs = [f(x) for x in sim]
    for _ in range(iters):
        o = np.argsort(fs); sim = [sim[i] for i in o]; fs = [fs[i] for i in o]
        c = np.mean(sim[:-1], axis=0)
        xr = c + (c - sim[-1]); fr = f(xr)
        if fr < fs[0]:
            xe = c + 2 * (c - sim[-1]); fe = f(xe)
            sim[-1], fs[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < fs[-2]:
            sim[-1], fs[-1] = xr, fr
        else:
            xc = c + 0.5 * (sim[-1] - c); fc = f(xc)
            if fc < fs[-1]:
                sim[-1], fs[-1] = xc, fc
            else:
                sim = [sim[0]] + [sim[0] + 0.5 * (x - sim[0]) for x in sim[1:]]
                fs = [fs[0]] + [f(x) for x in sim[1:]]
    i = int(np.argmin(fs))
    return sim[i], fs[i]


def fit_false_alarm(front, y):
    """ML of the scalar fa in P(report) = fa * front (front in (0,1], y a per-view report fraction): the Bernoulli log-likelihood is
    concave in fa, so a golden-section search on (0, 1/max(front)) finds it."""
    def f(fa):
        p = np.clip(fa * front, 1e-9, 1 - 1e-9)
        return -float(np.sum(y * np.log(p) + (1 - y) * np.log1p(-p)))
    lo, hi = 1e-6, min(0.999, 1.0 / float(front.max()) - 1e-6)
    g = (math.sqrt(5) - 1) / 2
    c, d = hi - g * (hi - lo), lo + g * (hi - lo)
    for _ in range(80):
        if f(c) < f(d): hi = d
        else: lo = c
        c, d = hi - g * (hi - lo), lo + g * (hi - lo)
    return 0.5 * (lo + hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="+"); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--fix-envelope", action="store_true")
    ap.add_argument("--out", default="etc/door_detectors.csv"); ap.add_argument("--source", default=None)
    a = ap.parse_args()
    rows = []
    for i, lg in enumerate(a.log):
        for r in csv.DictReader(open(lg, newline="")):
            r["door"] = f"{i}:{r['door']}"          # doors of different tapes are different doors
            rows.append(r)
    cols = rows[0].keys() if rows else []
    if rows and all(k in cols for k in ("door", "camera", "view_id")):   # one trial per view: the mean of its frames' rows
        groups = {}
        for r in rows:
            if r.get("q", "") in ("nan", "NaN"): continue                 # a ricoh frame with no model: q unknown, not a trial
            groups.setdefault((r["door"], r["camera"], r["view_id"]), []).append(r)
        rows = []
        for (door, cam, vid), g in groups.items():
            m = {k: sum(float(x[k]) for x in g) / len(g) for k in ("fill_h", "fill_v", "vis", "kappa") }
            if "aspect" in cols: m["aspect"] = sum(float(x["aspect"]) for x in g) / len(g)
            m["hit"] = sum(1.0 if float(x["hit"]) > 0 else 0.0 for x in g) / len(g)   # unweighted: the fraction of frames hit
            for k in ("alive_end", "crossed", "label"):
                if k in cols: m[k] = g[-1][k]
            m.update(door=door, camera=cam, view_id=vid)
            rows.append({k: str(v) for k, v in m.items()})
    have_label = "label" in cols
    have_filter = "alive_end" in cols and "crossed" in cols and not have_label
    neg_rows = [r for r in rows if have_label and float(r["label"]) == 0]
    if have_label:
        rows = [r for r in rows if float(r["label"]) == 1]
    if have_filter:
        rows = [r for r in rows if float(r["alive_end"]) > 0 and float(r["crossed"]) > 0]
    print(f"{len(rows)} views" + (" of true doors (label 1)" if have_label else
          " of doors alive at the end AND crossed" if have_filter else
          " (no alive_end/crossed columns: ALL doors taken as alive at the end, not filtered by crossing)")
          + (f"; {len(neg_rows)} views of non-doors (label 0)" if have_label else ""))
    fit = {}
    for cam in ("zed", "ricoh"):
        rr = [r for r in rows if r["camera"] == cam]
        if len(rr) < MIN_VIEWS:
            print(f"{cam}: {len(rr)} views, not fitted"); continue
        col = lambda k, dflt=None: np.array([float(r[k]) if k in r and r[k] != "" else dflt for r in rr])
        d = dict(fh=col("fill_h"), fv=col("fill_v"), asp=col("aspect", 1.0) if "aspect" in cols else np.ones(len(rr)),
                 vis=col("vis"), kap=col("kappa"), y=col("hit"))
        if a.fix_envelope:
            e = DEFAULT_ROW[cam]
            lo, hi, g = 1e-4, 1 - 1e-4, (math.sqrt(5) - 1) / 2   # nll is convex in hit_rate at a fixed front end: golden section
            c_, d_ = hi - g * (hi - lo), lo + g * (hi - lo)
            for _ in range(80):
                if nll((e[0], e[1], e[2], c_), d) < nll((e[0], e[1], e[2], d_), d): hi = d_
                else: lo = c_
                c_, d_ = hi - g * (hi - lo), lo + g * (hi - lo)
            x = np.array([e[0], e[1], e[2], 0.5 * (lo + hi)]); f = nll(tuple(x), d)
        else:
            x, f = nelder_mead(lambda p: nll(p, d), START[cam])
        fa, n_fa = None, 0
        nn = [r for r in neg_rows if r["camera"] == cam]
        if nn:
            colN = lambda k, dflt=None: np.array([float(r[k]) if k in r and r[k] != "" else dflt for r in nn])
            fN = np.array([float(a_) for a_ in (predict((x[0], x[1], x[2], 1.0), colN("fill_h"), colN("fill_v"),
                          colN("aspect", 1.0) if "aspect" in cols else np.ones(len(nn)), colN("vis"), colN("kappa")))])
            yN = colN("hit")
            use = fN > 1e-6
            if int(use.sum()) >= MIN_FA_VIEWS:
                fa = fit_false_alarm(fN[use], yN[use]); n_fa = int(use.sum())
                print(f"{cam}: false_alarm={fa:.4f} from {n_fa} label-0 views (mean front end {fN[use].mean():.3f}, "
                      f"mean report fraction {yN[use].mean():.4f}, {yN[use].sum():.1f} reports)")
        fit[cam] = (x, len(rr), float(d["y"].sum()), fa, n_fa)
        print(f"{cam}: n_views={len(rr)} hits={d['y'].sum():.1f} misses={len(rr)-d['y'].sum():.1f} "
              f"min_fill={x[0]:.4f} max_fill={x[1]:.4f} soft={x[2]:.4f} hit_rate={x[3]:.4f}  nll={f:.1f}")
    if a.dry_run:
        return 0
    src = a.source or f"fit_detector_envelopes.py {a.log} {datetime.date.today()}"
    # The other rows (support, confidence, geometry, lidar) are not fitted here: they are carried over from the file being replaced, so a refit
    # of the detector rates never silently drops them (the agent warns loudly on a missing row; see detector_llr.h).
    kept_rows = [ln for ln in open(a.out) if not ln.startswith(("zed,", "ricoh,", "camera,"))] if os.path.exists(a.out) else []
    with open(a.out, "w", newline="") as o:
        o.write("camera,param,value,n,source\n")
        for cam in ("zed", "ricoh"):
            if cam not in fit:   # not enough views: the compiled row, n = 0
                x, n, nf, fa, n_fa, src_c = DEFAULT_ROW[cam], 0, 0.0, None, 0, "compiled_defaults"
            else:
                x, n, _, fa, n_fa = fit[cam]; src_c = src
            if fa is None:
                fa, n_fa, src_fa = DEFAULT_FA[cam], 0, "compiled_defaults"
            else:
                src_fa = src
            for k, v, nn, sc in (("min_fill", x[0], 0, src_c), ("max_fill", x[1], 0, src_c), ("soft", x[2], 0, src_c),
                                 ("hit_rate", x[3], n, src_c), ("false_alarm", fa, n_fa, src_fa)):
                o.write(f"{cam},{k},{float(v):.6g},{nn},{sc}\n")
        o.writelines(kept_rows)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
