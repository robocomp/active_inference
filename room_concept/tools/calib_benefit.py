#!/usr/bin/env python3
"""Does self-calibration improve the prediction? A counterfactual replay on the SAME run, no ground truth.

    tools/calib_benefit.py tmp/heading/heading_<start>.csv [--mask -1]

Every logged prediction already contains the calibration corrections that were ACTING at the time. The
calibration is linear in its parameters, so the NOMINAL prediction (all parameters at their priors) is
recovered by subtracting, cycle by cycle, the correction each acting parameter added, scaled by the rest
gain exactly as the prediction was:
    heading  : k_omega*th_gyro - b_omega*t_gyro + k_omega_w*th_wheel + dk_wheel*fwd_wheel   (x rest gain ro)
    position : k_v/(1+k_v)*d_fwd*u_fwd + k_lat/(1+k_lat)*d_lat*u_lat - eps_yaw*d_fwd*u_lat   (x rest gain tr)
with u_fwd = (-sin th, cos th), u_lat = (cos th, sin th) (forward is +90 deg on this robot).

Scored between CONSECUTIVE LIDAR-SOLVED poses (iters > 0 at both ends), so both ends are anchored by the
room evidence, not by the prediction being judged:
    learned error = est_j - pred_j            (the correction the localiser applied)
    nominal error = that + the corrections summed over the stretch
The LiDAR pose is a noisy reference, but it is the SAME reference for both models, so the difference is the
calibration's effect. In simulation the same windows are also scored against ground truth (gt_*).

--mask: MotionCalibApplyMask the run used (1 = k_v only, -1 = all). Logs written before the calib_applied
column existed need it; newer logs carry the applied mask and ignore it.
"""
import argparse, math, os, sys
import numpy as np

P = ["k_v", "eps_yaw", "k_omega", "b_omega", "k_lat", "dk_wheel", "k_omega_w"]
W = lambda a: (a + np.pi) % (2 * np.pi) - np.pi


def load(path):
    L = [l for l in open(path) if not l.startswith("#")]
    h = L[0].strip().split(",")
    rows = [l.strip().split(",") for l in L[1:]]
    rows = [r for r in rows if len(r) == len(h)]
    a = np.array([[float(x) if x not in ("", "nan", "-nan", "inf", "-inf") else np.nan for x in r] for r in rows])
    return {k: a[:, i] for i, k in enumerate(h)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--mask", type=int, default=-1)
    a = ap.parse_args()
    for path in a.csv:
        d = load(path)
        n = len(d["ts_ms"])
        if "calib_applied" in d:
            applied = d["calib_applied"].astype(int)
        else:
            applied = d["calib_informed_mask"].astype(int) & (a.mask & 0x7F)
        act = {p: np.where((applied >> i) & 1, np.nan_to_num(d[f"v_{p}"]), 0.0) for i, p in enumerate(P)}
        g_ro = np.clip(np.nan_to_num(d["rest_gain_ro"], nan=1.0), 0, 1)
        g_tr = np.clip(np.nan_to_num(d["rest_gain_tr"], nan=1.0), 0, 1)
        g_ro[d["rest_gain_ro"] < 0] = 1.0
        g_tr[d["rest_gain_tr"] < 0] = 1.0
        # per-cycle correction the calibration ADDED to the prediction (world frame for position)
        dth_corr = g_ro * (act["k_omega"] * d["hc_th_gyro"] - act["b_omega"] * d["hc_t_gyro"]
                           + act["k_omega_w"] * d["hc_th_wheel"] + act["dk_wheel"] * d["hc_fwd_wheel"])
        th0 = np.r_[d["est_th"][0], d["est_th"][:-1]]          # heading the increment was integrated from
        uf = np.stack([-np.sin(th0), np.cos(th0)], 1)
        ul = np.stack([np.cos(th0), np.sin(th0)], 1)
        dfw, dlt = np.nan_to_num(d["dy_local"]), np.nan_to_num(d["dx_local"])
        kv, kl = act["k_v"], act["k_lat"]
        along = np.where(1 + kv != 0, kv / (1 + kv), 0) * dfw
        cross = np.where(1 + kl != 0, kl / (1 + kl), 0) * dlt - act["eps_yaw"] * dfw
        dpos_corr = g_tr[:, None] * (along[:, None] * uf + cross[:, None] * ul)

        solved = np.where(d["iters"] > 0)[0]
        est = np.stack([d["est_x"], d["est_y"]], 1); pred = np.stack([d["pred_x"], d["pred_y"]], 1)
        has_gt = np.isfinite(d["gt_x"]).sum() > 10
        acc = {k: [] for k in ("eth_l", "eth_n", "ep_l", "ep_n", "turn", "path", "dt", "gth_l", "gth_n")}
        t = d["ts_ms"] / 1000.0
        for i, j in zip(solved[:-1], solved[1:]):
            s = slice(i + 1, j + 1)
            c_th = W(d["est_th"][j] - d["pred_th"][j])
            c_p = est[j] - pred[j]
            sth, sp = dth_corr[s].sum(), dpos_corr[s].sum(0)
            acc["eth_l"].append(c_th); acc["eth_n"].append(c_th + sth)
            acc["ep_l"].append(c_p); acc["ep_n"].append(c_p + sp)
            acc["turn"].append(abs(W(d["est_th"][j] - d["est_th"][i])))
            acc["path"].append(np.linalg.norm(est[j] - est[i])); acc["dt"].append(t[j] - t[i])
            if has_gt:
                # truth increment vs predicted increment over the same stretch (predicted = pred_j - est_i)
                gth = W(d["gt_th"][j] - d["gt_th"][i])
                pth = W(d["pred_th"][j] - d["est_th"][i])
                acc["gth_l"].append(W(gth - pth)); acc["gth_n"].append(W(gth - (pth - sth)))
        A = {k: np.array(v) for k, v in acc.items()}
        if len(A["turn"]) == 0:
            print(f"{os.path.basename(path)}: no consecutive solved poses"); continue
        moving = A["path"] > 0.005
        turn, path_m, T = A["turn"].sum(), A["path"].sum(), A["dt"][moving].sum()
        print(f"\n══ {os.path.basename(path)}  — {len(A['turn'])} LiDAR-anchored stretches, {path_m:.1f} m, "
              f"{math.degrees(turn):.0f} deg turned; mask {'from log' if 'calib_applied' in d else a.mask}")
        for name, eth, ep in (("NOMINAL (no calibration)", A["eth_n"], A["ep_n"]),
                              ("LEARNED (as run)", A["eth_l"], A["ep_l"])):
            print(f"  {name:26s} heading |err| {math.degrees(np.abs(eth).sum()) / max(math.degrees(turn), 1e-9):.4f} deg/deg turned"
                  f" | signed drift while moving {math.degrees(eth[moving].sum()) / max(T, 1e-9):+.4f} deg/s"
                  f" | position |err| {1000 * np.linalg.norm(ep, axis=1).sum() / max(path_m, 1e-9):.2f} mm/m")
        dh = math.degrees(np.abs(A["eth_n"]).sum() - np.abs(A["eth_l"]).sum()) / max(math.degrees(turn), 1e-9)
        dp = 1000 * (np.linalg.norm(A["ep_n"], axis=1).sum() - np.linalg.norm(A["ep_l"], axis=1).sum()) / max(path_m, 1e-9)
        print(f"  BENEFIT of calibration     heading {dh:+.4f} deg/deg turned, position {dp:+.2f} mm/m"
              f"  (positive = calibration helped)")
        if has_gt:
            for name, e in (("NOMINAL", A["gth_n"]), ("LEARNED", A["gth_l"])):
                print(f"  vs GROUND TRUTH {name:8s} heading |err| {math.degrees(np.abs(e).sum()) / max(math.degrees(turn), 1e-9):.4f} deg/deg,"
                      f" signed drift while moving {math.degrees(e[moving].sum()) / max(T, 1e-9):+.4f} deg/s")


if __name__ == "__main__":
    main()
