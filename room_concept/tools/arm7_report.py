#!/usr/bin/env python3
"""Grade arm 7 (EXPERIMENT.md §13): all calibration parameters applied online vs forward scale only.

    tools/arm7_report.py etc/runs            (reads arm7_<leg>.csv, the heading logs saved per leg)

Per leg, all from the heading CSV (one row per localiser result):
  * CORRECTION LOAD  sum|est - pred| per metre driven (mm/m) -- the established endpoint (§6): what the
    optimiser had to remove from the motion model's guess; by thirds of the run, where online
    calibration has to show.
  * SURPRISE  KL nats/m and sum(mismatch)/sum(expected) (surprise.h), by thirds.
  * GROUND TRUTH (sim)  heading error of the PREDICTED increments against truth, deg per rad turned,
    and the translation scale error of the predicted path, over 5-s windows of motion.
  * the calibrator's final values +- sigma, and which parameters were informed.
"""
import math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from surprise_report import summary as sur_summary   # noqa: E402

LEGS = ["native-k", "native-all", "gyro-k", "gyro-all"]
PARAMS = ["k_v", "eps_yaw", "k_omega", "b_omega", "k_lat", "dk_wheel", "k_omega_w"]
W = lambda a: (a + np.pi) % (2 * np.pi) - np.pi


def load(path):
    L = [l for l in open(path) if not l.startswith("#")]
    h = L[0].strip().split(",")
    rows = [l.strip().split(",") for l in L[1:]]
    rows = [r for r in rows if len(r) == len(h)]
    a = np.array([[float(x) if x not in ("", "nan", "-nan", "inf", "-inf") else np.nan for x in r] for r in rows])
    return {k: a[:, i] for i, k in enumerate(h)}


def thirds(n):
    return [slice(k * n // 3, (k + 1) * n // 3) for k in range(3)]


def grade(d):
    t = (d["ts_ms"] - d["ts_ms"][0]) / 1000.0
    ex, ey, eth = d["est_x"], d["est_y"], d["est_th"]
    px, py, pth = d["pred_x"], d["pred_y"], d["pred_th"]
    corr = np.hypot(ex - px, ey - py)                        # what the optimiser removed, per row
    step = np.r_[0.0, np.hypot(np.diff(ex), np.diff(ey))]    # path driven, per row
    out = {"T": t[-1], "path": step.sum()}
    out["load"] = [1000 * corr[s].sum() / max(step[s].sum(), 1e-9) for s in thirds(len(t))]
    out["load_all"] = 1000 * corr.sum() / max(step.sum(), 1e-9)
    # ground truth: predicted increments vs true increments, 5-s windows with motion
    gx, gy, gth = d["gt_x"], d["gt_y"], d["gt_th"]
    out["gt"] = None
    if np.isfinite(gx).sum() > 10:
        dpth = np.r_[0.0, W(pth[1:] - eth[:-1])]
        dgth = np.r_[0.0, W(np.diff(gth))]
        dpp = np.r_[0.0, np.hypot(px[1:] - ex[:-1], py[1:] - ey[:-1])]
        dgp = np.r_[0.0, np.hypot(np.diff(gx), np.diff(gy))]
        win = (t // 5).astype(int)
        eh, turned, sp, sg = [], 0.0, 0.0, 0.0
        for b in np.unique(win):
            m = win == b
            if not np.isfinite(dgth[m]).all():
                continue
            eh.append(abs(dpth[m].sum() - dgth[m].sum())); turned += np.abs(dgth[m]).sum()
            sp += dpp[m].sum(); sg += dgp[m].sum()
        out["gt"] = dict(deg_per_rad=math.degrees(sum(eh)) / max(turned, 1e-9),
                         scale_err=100 * (sp - sg) / max(sg, 1e-9), turned=turned, gt_path=sg)
    last = {p: (d[f"v_{p}"][-1], d[f"s_{p}"][-1]) for p in PARAMS if f"v_{p}" in d}
    out["calib"] = last
    out["informed"] = int(d["calib_informed_mask"][-1]) if "calib_informed_mask" in d else -1
    return out


def sur(path):
    import surprise_report as sr
    rows = sr.load(path)
    if not rows:
        return None
    n = len(rows)
    return sr.summary(rows), [sr.summary(rows[s]) for s in thirds(n)]


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "etc", "runs")
    legs = [(l, os.path.join(d, f"arm7_{l}.csv")) for l in LEGS if os.path.exists(os.path.join(d, f"arm7_{l}.csv"))]
    if not legs:
        sys.exit(f"no arm7_<leg>.csv in {d}")
    res = {}
    for leg, p in legs:
        g = grade(load(p)); s = sur(p)
        res[leg] = (g, s)
        print(f"\n══ {leg}  ({os.path.basename(p)})  {g['T']:.0f} s, {g['path']:.1f} m ══")
        print("  correction load mm/m   all {:.1f}   by thirds {}".format(
            g["load_all"], "  ".join(f"{x:.1f}" for x in g["load"])))
        if s:
            a, th = s
            print("  surprise nats/m       all {:.3g}   by thirds {}".format(
                a["kl_m"], "  ".join(f"{x['kl_m']:.3g}" for x in th)))
            print("  mismatch/expected     all {:.2f} [{:.2f}, {:.2f}]   by thirds {}".format(
                a["ratio"], a["lo"], a["hi"], "  ".join(f"{x['ratio']:.2f}" for x in th)))
        if g["gt"]:
            G = g["gt"]
            print(f"  GT: heading {G['deg_per_rad']:.3f} deg/rad turned ({G['turned']:.0f} rad),"
                  f" translation scale {G['scale_err']:+.2f} %")
        print("  calibrator (final):  informed mask", g["informed"])
        for p_, (v, sd) in g["calib"].items():
            print(f"     {p_:10s} {v:+.5f} ± {sd:.5f}")
    print("\n══ PRE-REGISTERED COMPARISONS (EXPERIMENT.md §13) ══")
    for tag in ("native", "gyro"):
        k, a = res.get(f"{tag}-k"), res.get(f"{tag}-all")
        if not (k and a):
            print(f"  {tag}: leg pair incomplete"); continue
        lk, la = k[0]["load"][2], a[0]["load"][2]
        print(f"  {tag}: last-third correction load  mask1 {lk:.1f}  all {la:.1f} mm/m  ->  "
              f"{'ALL LOWER' if la < lk else 'ALL NOT LOWER'} ({100 * (la - lk) / lk:+.0f} %)")
        if k[1] and a[1]:
            print(f"  {tag}: last-third mismatch/expected  mask1 {k[1][1][2]['ratio']:.2f}  all {a[1][1][2]['ratio']:.2f}")
        if tag == "gyro":
            v, sd = a[0]["calib"].get("b_omega", (float("nan"), float("nan")))
            ok = abs(v - 0.002) <= 2 * sd if sd > 0 else False
            print(f"  gyro-all: b_omega {v:+.5f} ± {sd:.5f} vs injected +0.00200  ->  "
                  f"{'RECOVERED (within 2 sigma)' if ok else 'NOT RECOVERED'}")


if __name__ == "__main__":
    main()
