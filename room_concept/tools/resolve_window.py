#!/usr/bin/env python3
"""Re-solve a saved calibration window offline, under variant assumptions.

    resolve_window.py etc/motion_calib_state.csv [more_windows.csv ...]

WHY THIS EXISTS. The estimator is a batch least-squares over a window of episodes that is already
persisted to disk. So every question of the form "what would the estimator say if ..." can be
answered from the saved file in milliseconds, and only the GENERATION of episodes needs a robot.
On 2026-08-30 an entire day went into answering such questions by driving: does the fix
double-count, is the weighting biasing the slope, what would an unweighted fit give, does the
decomposition close. All of them are below, and all of them are instant.

★ Drive once, analyse many times. A hypothesis should cost seconds, not a 30-minute run.

WHAT IT REPRODUCES. The `shipped` row is the estimator's own arrangement — weighted by 1/pos_var,
prior at nominal, feedback undone via p_applied. If that row does not match what the agent
reported for the same window, the rest of the table is not a decomposition of anything and nothing
here should be believed. That check is printed first, and it is the only reason the rest is
readable as attribution.

THE ROWS, and what each isolates:
  shipped            what the estimator does now
  no prior           the prior's pull toward nominal
  unweighted         the observation weights (R = fit_model_gain * max|SDF| over the episode)
  no p_applied       the closed-loop feedback defect: covariates are POST-corrected odometry and an
                     episode records what was acting, so omitting it reads leftovers as totals
                     (fixed point v = k* S/(2S+P0): recovery capped at 50%)
  raw               all three off at once

FILE FORMAT. `E,d_forward,d_lateral,d_theta,duration,r_forward,r_lateral,r_theta,pos_var,theta_var`
plus, since 2026-08-30, six p_applied columns. A 9-column file predates that and is read with
p_applied = 0 -- which is exactly the defective reading, so an old file will show `shipped` and
`no p_applied` agreeing. That is not a null result; it means the file cannot answer the question.
"""
import csv
import math
import sys

import numpy as np

P_NAMES = ["k_v", "eps_yaw", "k_omega", "b_omega", "k_lat", "dk_wheel"]
P_COUNT = len(P_NAMES)
# Prior sigmas, from rc::calib::Prior. Keep in step with calibration_estimator.h.
PRIOR_SIGMA = np.array([0.02, 0.0175, 0.02, 5.0e-4, 0.05, 0.02])


def load(path):
    """Episodes as arrays. Returns (E, has_p_applied)."""
    rows = []
    with open(path) as f:
        for line in f:
            if not line.startswith("E,"):
                continue
            try:
                v = [float(x) for x in line.strip().split(",")[1:]]
            except ValueError:
                continue
            if len(v) in (9, 9 + P_COUNT):
                rows.append(v + [0.0] * (9 + P_COUNT - len(v)))
    if not rows:
        sys.exit(f"{path}: no episode rows")
    a = np.asarray(rows)
    # A file written before p_applied existed has it all-zero; so does a genuinely
    # never-corrected window. Distinguish by column count on the first row.
    with open(path) as f:
        widths = {len(l.strip().split(",")) - 1 for l in f if l.startswith("E,")}
    return a, (max(widths) == 9 + P_COUNT)


def solve(a, *, weighted=True, prior=True, undo_feedback=True):
    """The estimator's normal equations, with each term switchable.

    Mirrors BatchEstimator::solve -- three rows per episode, sharing one parameter vector:
      along  : k_v                        on d_forward
      cross  : eps_yaw, k_lat             on -d_forward, d_lateral
      heading: k_omega, b_omega, dk_wheel on d_theta, duration, d_forward
    """
    d_f, d_l, d_th, dur = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    r_f, r_l, r_th = a[:, 4], a[:, 5], a[:, 6]
    pos_var, th_var = a[:, 7], a[:, 8]
    p_app = a[:, 9:9 + P_COUNT]

    H = np.zeros((P_COUNT, P_COUNT))
    b = np.zeros(P_COUNT)
    for i in range(len(a)):
        wp = 1.0 / max(pos_var[i], 1e-12) if weighted else 1.0
        wt = 1.0 / max(th_var[i], 1e-12) if weighted else 1.0
        for w, j, r in (
            (wp, np.array([d_f[i], 0, 0, 0, 0, 0]), r_f[i]),
            (wp, np.array([0, -d_f[i], 0, 0, d_l[i], 0]), r_l[i]),
            (wt, np.array([0, 0, d_th[i], dur[i], 0, d_f[i]]), r_th[i]),
        ):
            # Undo the feedback: the recorded residual is what remained AFTER p_applied acted.
            total = r + (j @ p_app[i] if undo_feedback else 0.0)
            H += w * np.outer(j, j)
            b += w * j * total
    if prior:
        H += np.diag(1.0 / PRIOR_SIGMA ** 2)
    # Normalised condition number, as the agent reports it: D^-1/2 H D^-1/2.
    dg = np.sqrt(np.maximum(np.diag(H), 1e-30))
    Hn = H / np.outer(dg, dg)
    cond = np.linalg.cond(Hn)
    try:
        p = np.linalg.solve(H + 1e-12 * np.eye(P_COUNT), b)
        sig = np.sqrt(np.maximum(np.diag(np.linalg.inv(H + 1e-12 * np.eye(P_COUNT))), 0.0))
    except np.linalg.LinAlgError:
        return None, None, cond
    return p, sig, cond


def report(path):
    a, has_pa = load(path)
    print(f"\n=== {path}")
    print(f"{len(a)} episodes | p_applied column present: {has_pa}")
    if not has_pa:
        print("  ⚠ 9-column file: written before p_applied existed. It is read with p_applied = 0,")
        print("    which IS the defective reading, so `shipped` and `no p_applied` will agree here")
        print("    for a reason that has nothing to do with the estimator. Re-record before using.")
    mean_pa = a[:, 9:9 + P_COUNT].mean(axis=0)
    if np.any(np.abs(mean_pa) > 1e-9):
        print("  mean p_applied acting over the window: "
              + ", ".join(f"{P_NAMES[i]} {mean_pa[i]:+.5f}" for i in range(P_COUNT)
                          if abs(mean_pa[i]) > 1e-9))

    variants = [
        ("shipped",       dict(weighted=True,  prior=True,  undo_feedback=True)),
        ("no prior",      dict(weighted=True,  prior=False, undo_feedback=True)),
        ("unweighted",    dict(weighted=False, prior=True,  undo_feedback=True)),
        ("no p_applied",  dict(weighted=True,  prior=True,  undo_feedback=False)),
        ("raw",           dict(weighted=False, prior=False, undo_feedback=False)),
    ]
    print(f"\n{'variant':<14}" + "".join(f"{n:>12}" for n in P_NAMES) + f"{'cond':>8}")
    base = None
    for name, kw in variants:
        p, _sig, cond = solve(a, **kw)
        if p is None:
            print(f"{name:<14} singular")
            continue
        if base is None:
            base = p
        print(f"{name:<14}" + "".join(f"{p[i]:>12.6f}" for i in range(P_COUNT)) + f"{cond:>8.2f}")

    # The two questions the variants exist to answer, stated rather than left to the reader.
    p_ship, _, _ = solve(a)
    p_nofb, _, _ = solve(a, undo_feedback=False)
    p_unw, _, _ = solve(a, weighted=False)
    if p_ship is not None and p_nofb is not None and abs(p_ship[0]) > 1e-9:
        print(f"\nfeedback term is worth  {100 * (1 - p_nofb[0] / p_ship[0]):+.1f}% of k_v "
              f"({p_nofb[0]:+.6f} without it, {p_ship[0]:+.6f} with)")
    if p_ship is not None and p_unw is not None and abs(p_ship[0]) > 1e-9:
        print(f"weighting term is worth {100 * (1 - p_ship[0] / p_unw[0]):+.1f}% of k_v "
              f"({p_unw[0]:+.6f} unweighted, {p_ship[0]:+.6f} weighted)")

    # Selection-on-outcome check: R should not grow with the size of the response.
    r_f, pos_var = a[:, 4], a[:, 7]
    m = np.isfinite(r_f) & np.isfinite(pos_var) & (pos_var > 0)
    if m.sum() > 10:
        c = np.corrcoef(np.abs(r_f[m]), pos_var[m])[0, 1]
        print(f"corr(|r_forward|, pos_var) = {c:+.3f}"
              + ("   ⚠ positive: the biggest true errors are trusted least" if c > 0.1 else ""))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        report(p)
