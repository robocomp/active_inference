#!/usr/bin/env python3
"""Does applying the self-calibrated motion parameters improve LOCALIZATION?

usage: calib_localization_ab.py A.csv "label A" B.csv "label B" [...]
       (any number of arms; each is one tmp/sdf_localizer/gt_error.csv)

WHAT THIS ADDS OVER run_compare.py
----------------------------------
`run_compare.py` grades the PREDICTOR (err/m, ramp steepness, opt/m) — the area under the
sawtooth. That is the right instrument for "is the motion model better", and it is what the
2026-08-22 A/B used to conclude the calibrator lowers the error FLOOR by 14.9%.

It is NOT the instrument for "is the ROBOT BETTER LOCALISED", because it never looks at the
published pose against ground truth. This script does exactly that, and nothing else.

★★★ THE TRAP THIS SCRIPT EXISTS TO AVOID: THE TWO POSES ARE IN DIFFERENT FRAMES.
GT is the Webots world frame; the estimate is the ROOM frame, whose orientation room_concept
picks from its own fit and which is free to differ between runs (it has been observed sitting in
stable modes tens of degrees apart, and a constant offset carries NO information — only its
stability does). So `|est - gt|` is NOT a localisation error: most of it is the frame choice, and
it changes when nothing about the localiser changed. Comparing two arms on that number compares
two arbitrary frame choices. See gt_analyze.py's header and [[comparing-unaligned-measurements]].

So the PRIMARY metric here is RELATIVE pose error (RPE), which is invariant to the frame
entirely: over a fixed span of GT travel, how far does the estimate's own motion differ from the
true motion? No alignment, no offset, nothing to get wrong.

    for a pair (i, j) spanning ds metres of GT path:
        rel_gt  = inv(T_gt_i)  @ T_gt_j        the true motion over that span
        rel_est = inv(T_est_i) @ T_est_j       what the localiser thinks it did
        e_trans = || rel_gt.xy - rel_est.xy ||     reported in mm per metre travelled
        e_rot   = wrap(rel_gt.th - rel_est.th)     reported in deg per radian turned

This is also the metric the calibration can actually move: k_v, k_omega and eps_yaw are motion-
model parameters, so they act on incremental motion, which is precisely what RPE measures.

Aligned ATE is reported as a SECONDARY, after fitting one rigid SE(2) per window (Umeyama, no
scale). It is a consistency figure, not an accuracy one — read it as "how well does a single
rigid transform explain this window", never as "how far the robot was from where it thought".

PRE-REGISTER BEFORE YOU RUN (print this, then run):
  - endpoint            : RPE translation (mm/m) at ds = 1.0 m, MOVING rows only
  - secondary           : RPE rotation (deg/rad) at dth = 0.5 rad; aligned ATE RMS
  - unit of analysis    : 60 s window; arms compared window-to-window
  - exclusions          : parked rows; burst windows (>50% of cycles with iters>0)
  - direction predicted : calibration ON <= OFF on both RPE channels
  - expected size       : SMALL on a healthy robot (k_v 1.00004, k_omega 0.9974 as of 2026-08-23);
                          eps_yaw (-0.536 deg) is the only sizeable term. Run the INJECTED-ERROR
                          arm if you need power — see EXPERIMENT_CALIB_LOCALIZATION.md.

★ A null result on the healthy robot is a legitimate answer, not a failed experiment — but only
  if the injected-error arm shows the same pipeline CAN detect an effect. Report both or neither.
"""
import csv
import math
import sys

import numpy as np

# ── knobs that are DEFINITIONS, not tuning ────────────────────────────────────────────────────
DS_TRANS = 1.0      # m of GT path per RPE translation sample
DTH_ROT = 0.5       # rad of GT rotation per RPE rotation sample
WINDOW_S = 60.0     # unit of analysis
PARKED_MPS = 0.02   # below this GT speed the robot is parked and predicts nothing: a parked
                    # robot always looks accurate, and pooling it with moving rows is how a
                    # confident false finding gets made.
BURST_FRAC = 0.50   # a window with more than this fraction of optimised cycles is the known
                    # pathological regime (the localiser is not tracking); it smears any average
                    # it lands in and is reported separately, never mixed.


# ★★★ gt_theta AND est_theta DO NOT SHARE A CONVENTION, AND THE CSV DOES NOT SAY SO.
# Measured 2026-08-29 on gt_error_2026-08-29_21-54-45.csv, as mean(course - theta) over all
# samples with real motion, where `course` is atan2 of the trajectory's own displacement:
#
#     GT   +0.01 deg  (concentration R = 1.000)   supervisor heading: forward = +x
#     EST  +89.77 deg (concentration R = 0.904)   room frame: forward = +y (the BODY frame,
#                                                 ROBOT_GEOMETRY.md / FRAMES.md)
#
# So est_theta must be turned by +90 deg to mean the same thing as gt_theta. This is a DECLARED
# convention taken from the frame documentation and confirmed against the data -- it is NOT
# fitted, and it must never become fitted: a free constant angular offset is exactly the shape of
# eps_yaw (-0.536 deg), the mount-yaw parameter this experiment is trying to measure. Fitting the
# offset would absorb the very effect under test and guarantee a null result.
#
# The 0.23 deg by which the measurement misses a clean 90 is therefore LEFT IN, as signal.
EST_YAW_OFFSET_DEG = 90.0
CONVENTION_TOL_DEG = 5.0   # a check, not a knob: past this the assumption is wrong, so STOP


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def course_minus_theta(x, y, th, min_step=0.005):
    """Circular mean of (direction of travel - reported heading), and its concentration R.

    R near 1 means the reported heading tracks the direction of travel; a low R on a HOLONOMIC
    base is not necessarily wrong (it can strafe), which is why this is reported and checked
    rather than used to correct anything.
    """
    dx, dy = np.diff(x), np.diff(y)
    m = np.hypot(dx, dy) > min_step
    if m.sum() < 20:
        return math.nan, 0.0, int(m.sum())
    off = wrap(np.arctan2(dy[m], dx[m]) - th[:-1][m])
    s, c = np.sin(off).mean(), np.cos(off).mean()
    return math.atan2(s, c), math.hypot(s, c), int(m.sum())


def load(path):
    """Rows as a dict of arrays, GT interpolated onto the estimate's timestamps.

    GT arrives from the supervisor at its own (slower) rate, so the column is a STAIRCASE. Using
    it raw puts a sawtooth into every error at the GT publication period, which looks exactly
    like a localiser artefact. Interpolate between the instants GT actually CHANGED.
    """
    cols = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            for k, v in r.items():
                if k is None:
                    continue
                try:
                    cols.setdefault(k, []).append(float(v))
                except (TypeError, ValueError):
                    cols.setdefault(k, []).append(math.nan)
    if not cols:
        sys.exit("%s: no rows" % path)
    d = {k: np.asarray(v, dtype=float) for k, v in cols.items()}

    need = ('ts_ms', 'gt_x', 'gt_y', 'gt_theta', 'est_x', 'est_y', 'est_theta')
    missing = [k for k in need if k not in d]
    if missing:
        sys.exit("%s: missing column(s) %s" % (path, ', '.join(missing)))

    ok = np.isfinite(d['ts_ms'])
    for k in need:
        ok &= np.isfinite(d[k])
    d = {k: v[ok] for k, v in d.items()}
    order = np.argsort(d['ts_ms'])
    d = {k: v[order] for k, v in d.items()}
    if len(d['ts_ms']) < 100:
        sys.exit("%s: only %d usable rows -- drive around first" % (path, len(d['ts_ms'])))

    t = (d['ts_ms'] - d['ts_ms'][0]) / 1000.0
    gx, gy, gth = d['gt_x'], d['gt_y'], np.unwrap(d['gt_theta'])
    chg = np.r_[True, (np.diff(gx) != 0) | (np.diff(gy) != 0) | (np.diff(d['gt_theta']) != 0)]
    if chg.sum() < 10:
        sys.exit("%s: ground truth never changes -- is the supervisor publishing robot_gt_*?" % path)
    d['t'] = t
    d['gx'] = np.interp(t, t[chg], gx[chg])
    d['gy'] = np.interp(t, t[chg], gy[chg])
    d['gth'] = np.interp(t, t[chg], gth[chg])
    d['ex'], d['ey'] = d['est_x'], d['est_y']
    d['eth'] = np.unwrap(d['est_theta']) + math.radians(EST_YAW_OFFSET_DEG)

    # Verify the convention on THIS file rather than trusting the constant. If the two columns
    # ever start agreeing (or disagree by something else), every number below would be wrong in a
    # way that looks like a localisation result, so this refuses rather than warns.
    og, rg, ng = course_minus_theta(d['gx'], d['gy'], d['gt_theta'])
    oe, re_, ne = course_minus_theta(d['ex'], d['ey'], d['est_theta'])
    d['_conv'] = (math.degrees(og), rg, ng, math.degrees(oe), re_, ne)
    if ne >= 20 and math.isfinite(oe):
        miss = abs(math.degrees(wrap(oe - math.radians(EST_YAW_OFFSET_DEG))))
        if miss > CONVENTION_TOL_DEG:
            sys.exit("%s: est heading convention is %+.2f deg off the direction of travel, not the "
                     "declared %+.1f (miss %.2f deg > %.1f). The +90 assumption does not hold for "
                     "this file -- fix the assumption, do NOT widen the tolerance."
                     % (path, math.degrees(oe), EST_YAW_OFFSET_DEG, miss, CONVENTION_TOL_DEG))
    for k in ('pred_x', 'pred_y', 'pred_theta'):
        if k not in d:
            d[k] = np.full_like(t, math.nan)
    d['px'], d['py'] = d['pred_x'], d['pred_y']
    d['pth'] = np.unwrap(np.nan_to_num(d['pred_theta'], nan=0.0)) if np.isfinite(d['pred_theta']).any() else d['pred_theta']
    if 'iters' not in d:
        d['iters'] = np.zeros_like(t)

    # GT path length and speed, from GT alone: the estimate must never define its own yardstick.
    step = np.hypot(np.diff(d['gx']), np.diff(d['gy']))
    d['s'] = np.r_[0.0, np.cumsum(step)]
    d['rot'] = np.r_[0.0, np.cumsum(np.abs(np.diff(d['gth'])))]
    dt = np.diff(t)
    spd = np.r_[0.0, np.where(dt > 1e-6, step / np.maximum(dt, 1e-6), 0.0)]
    # A short median filter: a single dropped GT sample makes one enormous spurious speed.
    d['speed'] = np.r_[spd[0], np.median(np.stack([spd[:-2], spd[1:-1], spd[2:]]), axis=0), spd[-1]]
    return d


def rpe(d, idx, span, which='trans'):
    """Relative pose error over a fixed span of GT motion. Frame-invariant by construction.

    Returns (translation error / metre, rotation error / radian) over the pairs found inside
    `idx`. Pairs are non-overlapping so the samples are independent -- overlapping pairs would
    reuse the same motion many times and make any t-test claim a sample size it does not have.
    """
    key = d['s'] if which == 'trans' else d['rot']
    sel = np.asarray(idx)
    if sel.size < 2:
        return np.array([]), np.array([])
    et, er = [], []
    i = 0
    while i < sel.size - 1:
        a = sel[i]
        j = np.searchsorted(key[sel], key[a] + span)
        if j >= sel.size:
            break
        b = sel[j]
        cg, sg = math.cos(d['gth'][a]), math.sin(d['gth'][a])
        ce, se = math.cos(d['eth'][a]), math.sin(d['eth'][a])
        dxg, dyg = d['gx'][b] - d['gx'][a], d['gy'][b] - d['gy'][a]
        dxe, dye = d['ex'][b] - d['ex'][a], d['ey'][b] - d['ey'][a]
        # into each trajectory's OWN body frame at the start of the pair
        gx_, gy_ = cg * dxg + sg * dyg, -sg * dxg + cg * dyg
        ex_, ey_ = ce * dxe + se * dye, -se * dxe + ce * dye
        et.append(math.hypot(gx_ - ex_, gy_ - ey_))
        er.append(abs(wrap((d['gth'][b] - d['gth'][a]) - (d['eth'][b] - d['eth'][a]))))
        i = j
    if not et:
        return np.array([]), np.array([])
    return np.asarray(et) / span, np.asarray(er) / span


def aligned_ate(d, idx):
    """RMS residual after fitting ONE rigid SE(2) over the window (Umeyama without scale).

    Secondary and clearly labelled: it removes the arbitrary room-frame offset, so what is left
    is how well a single rigid transform explains the window -- consistency, not accuracy.
    """
    sel = np.asarray(idx)
    if sel.size < 10:
        return math.nan
    P = np.stack([d['ex'][sel], d['ey'][sel]])
    Q = np.stack([d['gx'][sel], d['gy'][sel]])
    pc, qc = P.mean(axis=1, keepdims=True), Q.mean(axis=1, keepdims=True)
    H = (P - pc) @ (Q - qc).T
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, np.linalg.det(Vt.T @ U.T)])
    R = Vt.T @ D @ U.T
    res = (Q - qc) - R @ (P - pc)
    return float(np.sqrt((res ** 2).sum(axis=0).mean()))


def windows(d):
    """Yield (label, index array) per WINDOW_S, tagging parked-only and burst windows."""
    t = d['t']
    for w0 in np.arange(0.0, t[-1], WINDOW_S):
        idx = np.flatnonzero((t >= w0) & (t < w0 + WINDOW_S))
        if idx.size < 30:
            continue
        moving = idx[d['speed'][idx] >= PARKED_MPS]
        burst = float(np.mean(d['iters'][idx] > 0)) > BURST_FRAC
        yield w0, idx, moving, burst


def arm(path, label):
    d = load(path)
    rows = {'label': label, 'path': path, 'trans': [], 'rot': [], 'ate': [],
            'burst': 0, 'parked': 0, 'used': 0, 'dist': 0.0}
    for _w0, idx, moving, burst in windows(d):
        if burst:
            rows['burst'] += 1
            continue
        if moving.size < 30:
            rows['parked'] += 1
            continue
        et, _ = rpe(d, moving, DS_TRANS, 'trans')
        _, er = rpe(d, moving, DTH_ROT, 'rot')
        if et.size:
            rows['trans'].append(float(np.mean(et)) * 1000.0)      # mm per metre
        if er.size:
            rows['rot'].append(math.degrees(float(np.mean(er))))   # deg per radian
        a = aligned_ate(d, moving)
        if math.isfinite(a):
            rows['ate'].append(a * 1000.0)                          # mm
        rows['used'] += 1
        rows['dist'] += float(d['s'][moving[-1]] - d['s'][moving[0]])
    # what the calibrator actually held during the arm -- a result is only attributable to a
    # parameter that MOVED, and these columns are the only record of whether one did.
    rows['conv'] = d['_conv']
    for k, name in (('calib_k_v', 'k_v'), ('calib_k_w', 'k_omega'), ('calib_yaw', 'eps_yaw')):
        if k in d and np.isfinite(d[k]).any():
            v = d[k][np.isfinite(d[k])]
            rows[name] = (float(v[0]), float(v[-1]))
    return rows


def welch(a, b):
    """Welch t and a rank test, on the WINDOW means. Two tests because they fail differently:
    t is sensitive to a shifted mean, the rank test to a shifted distribution with outliers."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 3 or b.size < 3:
        return math.nan, math.nan, math.nan
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(va / a.size + vb / b.size)
    t = (a.mean() - b.mean()) / se if se > 0 else math.nan
    # Mann-Whitney U, normal approximation (window counts here are small but not tiny)
    allv = np.concatenate([a, b])
    r = np.argsort(np.argsort(allv)) + 1.0
    ua = r[:a.size].sum() - a.size * (a.size + 1) / 2.0
    mu = a.size * b.size / 2.0
    sd = math.sqrt(a.size * b.size * (a.size + b.size + 1) / 12.0)
    z = (ua - mu) / sd if sd > 0 else math.nan
    # Cohen's d on the pooled SD -- the effect SIZE, which is what a small-n study must report
    sp = math.sqrt(((a.size - 1) * va + (b.size - 1) * vb) / max(1, a.size + b.size - 2))
    dd = (a.mean() - b.mean()) / sp if sp > 0 else math.nan
    return t, z, dd


def main():
    if len(sys.argv) < 3 or len(sys.argv) % 2 == 0:
        sys.exit(__doc__)
    print(__doc__.split('PRE-REGISTER')[1].split('"""')[0].join(['PRE-REGISTER', '']).strip())
    print()
    arms = [arm(sys.argv[i], sys.argv[i + 1]) for i in range(1, len(sys.argv), 2)]

    print("%-22s %7s %8s %10s %10s %10s" %
          ("arm", "wins", "dist_m", "RPE mm/m", "RPE d/rad", "ATE mm"))
    for a in arms:
        print("%-22s %7d %8.1f %10.3f %10.3f %10.1f  (burst %d, parked %d)" %
              (a['label'], a['used'], a['dist'],
               np.mean(a['trans']) if a['trans'] else math.nan,
               np.mean(a['rot']) if a['rot'] else math.nan,
               np.mean(a['ate']) if a['ate'] else math.nan,
               a['burst'], a['parked']))

    print("\nheading convention check, mean(course - theta) per trajectory:")
    for a in arms:
        og, rg, ng, oe, re_, ne = a['conv']
        print("  %-20s GT %+7.2f deg (R %.3f, n %5d)   EST %+7.2f deg (R %.3f, n %5d)"
              % (a['label'], og, rg, ng, oe, re_, ne))
    print("  ^ EST is expected near %+.0f deg: the two columns use DIFFERENT conventions and the"
          % EST_YAW_OFFSET_DEG)
    print("    CSV does not say so. The offset is applied from the frame docs, never fitted --")
    print("    fitting it would absorb eps_yaw, the parameter under test.")

    print("\ncalibration state during each arm (first -> last):")
    for a in arms:
        bits = ["%s %.5f->%.5f" % (n, *a[n]) for n in ('k_v', 'k_omega', 'eps_yaw') if n in a]
        print("  %-20s %s" % (a['label'], '  '.join(bits) if bits else "(no calib_* columns)"))
    print("  ^ a difference between arms is only attributable to calibration if a parameter MOVED.")

    if len(arms) >= 2:
        print("\npairwise, on the per-window means (negative t = first arm LOWER error = better):")
        print("%-30s %10s %8s %8s %8s" % ("comparison", "metric", "t", "z", "d"))
        for i in range(len(arms)):
            for j in range(i + 1, len(arms)):
                A, B = arms[i], arms[j]
                for m, name in (('trans', 'RPE mm/m'), ('rot', 'RPE d/rad'), ('ate', 'ATE mm')):
                    t, z, dd = welch(A[m], B[m])
                    print("%-30s %10s %8.2f %8.2f %8.2f" %
                          ("%s vs %s" % (A['label'], B['label']), name, t, z, dd))
        print("\n  n is the WINDOW count, not the row count. With single-digit windows nothing")
        print("  here is significant whatever the t says -- extend the run, do not read harder.")


if __name__ == '__main__':
    main()
