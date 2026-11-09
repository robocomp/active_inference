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
false_alarm is not identifiable from views of real doors and is kept at the compiled default.
Usage: python3 -I tools/fit_detector_envelopes.py LOG.csv [--dry-run] [--out etc/door_detectors.csv] [--source TAG]
Writes the table only without --dry-run.
"""
import argparse, csv, datetime, os, sys
import numpy as np

DEFAULT_FA = {"zed": 0.02, "ricoh": 0.05}
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log"); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="etc/door_detectors.csv"); ap.add_argument("--source", default=None)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.log, newline="")))
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
            for k in ("alive_end", "crossed"):
                if k in cols: m[k] = g[-1][k]
            m.update(door=door, camera=cam, view_id=vid)
            rows.append({k: str(v) for k, v in m.items()})
    have_filter = "alive_end" in cols and "crossed" in cols
    if have_filter:
        rows = [r for r in rows if float(r["alive_end"]) > 0 and float(r["crossed"]) > 0]
    print(f"{len(rows)} views" + (" of doors alive at the end AND crossed" if have_filter else
          " (no alive_end/crossed columns: ALL doors taken as alive at the end, not filtered by crossing)"))
    fit = {}
    for cam in ("zed", "ricoh"):
        rr = [r for r in rows if r["camera"] == cam]
        if len(rr) < 20:
            print(f"{cam}: {len(rr)} views, not fitted"); continue
        col = lambda k, dflt=None: np.array([float(r[k]) if k in r and r[k] != "" else dflt for r in rr])
        d = dict(fh=col("fill_h"), fv=col("fill_v"), asp=col("aspect", 1.0) if "aspect" in cols else np.ones(len(rr)),
                 vis=col("vis"), kap=col("kappa"), y=col("hit"))
        x, f = nelder_mead(lambda p: nll(p, d), START[cam])
        fit[cam] = (x, len(rr), float(d["y"].sum()))
        print(f"{cam}: n_views={len(rr)} hits={d['y'].sum():.1f} misses={len(rr)-d['y'].sum():.1f} "
              f"min_fill={x[0]:.4f} max_fill={x[1]:.4f} soft={x[2]:.4f} hit_rate={x[3]:.4f}  nll={f:.1f}")
    if a.dry_run:
        return 0
    src = a.source or f"fit_detector_envelopes.py {a.log} {datetime.date.today()}"
    # The fit-support rows (camera `support`) are not fitted here: they are carried over from the file being replaced, so a refit
    # of the detector rates never silently drops them (the agent warns loudly on a missing row; see detector_llr.h).
    support_rows = [ln for ln in open(a.out) if ln.startswith(("support,", "confidence,"))] if os.path.exists(a.out) else []
    with open(a.out, "w", newline="") as o:
        o.write("camera,param,value,n,source\n")
        for cam in ("zed", "ricoh"):
            if cam not in fit:
                continue
            x, n, _ = fit[cam]
            for k, v, nn in (("min_fill", x[0], 0), ("max_fill", x[1], 0), ("soft", x[2], 0), ("hit_rate", x[3], n),
                             ("false_alarm", DEFAULT_FA[cam], 0)):
                o.write(f"{cam},{k},{float(v):.6g},{nn},{src}\n")
        o.writelines(support_rows)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
