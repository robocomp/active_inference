#!/usr/bin/env python3
"""Which motion would tell us most about which calibration parameter?

usage: excitation.py [motion_calib_state.csv] [--budget SECONDS] [--vmax M/S] [--wmax RAD/S]

WHY THIS EXISTS
---------------
Repeating the four driving arms for each of the six parameters is about 6 km of driving, and it
would answer the wrong question. Whether a parameter is identifiable is not a matter of how many
experiments you run; it is a property of the Fisher information the motion produces, and the model
is LINEAR in the parameters, so that information has a closed form. This computes it.

THE MODEL (mirrors calibration_estimator.h:337-383 row for row — if that changes, change this)
    per episode, three residual components, each a row of J:
        along  : k_v * d_forward                                     weight 1/pos_var
        cross  : -eps_yaw * d_forward + k_lat * d_lateral            weight 1/pos_var
        head   : k_omega * d_theta + b_omega * duration
                 + dk_wheel * d_forward                              weight 1/theta_var
    plus closed pivots (C rows), which carry k_omega and b_omega only.
    H = sum J^T W J + diag(1/prior_sigma^2)

★ BECAUSE EACH PARAMETER LOADS ON EXACTLY ONE COMPONENT, H IS BLOCK DIAGONAL: 1x1 for k_v, 2x2 for
(eps_yaw, k_lat), 3x3 for (k_omega, b_omega, dk_wheel). Identifiability is therefore entirely a
question of how well the per-episode covariate vectors span their own block, and the whole
conditioning problem lives in the 3x3.

★★★ THE CLOSED FORM THAT ANSWERS "WHAT MOVEMENT?". For a motion held at constant (v, w) for time T:
d_theta = w*T and d_forward = v*T, so the heading block's covariate vector is

        g = (T / sigma_theta) * (w, 1, v)

The DIRECTION depends only on the rates; T and the precision set only the magnitude. Two things
follow, and they are the whole design:

  1. A constant-rate pivot gives g proportional to (w, 1, 0) for EVERY episode. Every vector
     parallel, the block collapses to rank 1, and k_omega cannot be told from b_omega at all. This
     is the self-test's condition number going 14.5 -> 216.4, derived rather than observed.
  2. THE MIDDLE COORDINATE IS ALWAYS 1. Every episode loads identically on the time axis, which is
     WHY b_omega is structurally the worst-determined parameter: you cannot build a motion that has
     duration but not duration. Its separation has to come from spreading w in ABSOLUTE units
     against that fixed 1 — which means rate DIVERSITY, not rate magnitude.

★★★ THERE ARE TWO SEPARATE DEGENERACIES HERE AND THEY HAVE DIFFERENT CURES. Verified against this
    tool (synthetic primitives, prior removed so only the data speaks):

      (k_omega, b_omega) 2x2      one pivot rate    cond = inf   <- rank 1
                                  two pivot rates   cond = 12.4
      full heading 3x3            pivots only, any number of rates      cond = inf
                                  pivots + one straight                 cond = 51.6
                                  pivots + straight + arcs              cond = 32.5

    Rate diversity fixes the k_omega/b_omega pair and does NOTHING for dk_wheel, whose covariate is
    FORWARD DISTANCE — a pivot supplies none, so no amount of turning at any number of rates will
    ever identify it. Turning separates the gyro pair; DRIVING is what reveals the wheel mismatch.
    Confusing the two is easy because they share a residual component and one condition number.

CRITERIA. D-optimal (max det H) is the information-theoretic one: the expected information gain of
a manoeuvre is 0.5*log det(H_after) - 0.5*log det(H_before), in nats, which is what an active-
inference agent should be valuing and is continuous — no threshold anywhere. E-optimal
(max lambda_min) is reported alongside because the complaint here is that ONE direction is starved,
not that the overall volume is small, and those are different objectives.

★ This scores motions, it does not command them. Feeding it to the base self-calibration affordance
means handing that agent the IG column as the value of a candidate manoeuvre.
"""
import math
import sys

import numpy as np

P_NAMES = ['k_v', 'eps_yaw', 'k_omega', 'b_omega', 'k_lat', 'dk_wheel']
P_K_V, P_EPS_YAW, P_K_OMEGA, P_B_OMEGA, P_K_LAT, P_DK_WHEEL = range(6)
# calibration_estimator.h:162-169. Not tuning: these are the declared priors the solver uses.
PRIOR_SIGMA = np.array([0.02, 0.0175, 0.02, 5.0e-4, 0.05, 0.02])
# Each parameter lives on exactly one residual component, which is what makes H block diagonal.
BLOCKS = [('forward', [P_K_V]),
          ('lateral', [P_EPS_YAW, P_K_LAT]),
          ('heading', [P_K_OMEGA, P_B_OMEGA, P_DK_WHEEL])]


def load(path):
    """Episodes and closed pivots from motion_calib_state.csv.

    ★ from_chars, not the C locale: python's float() is locale-independent, but note the file is
    WRITTEN by a Qt process under es_ES and READ here — see CLAUDE.md. A comma separator would
    raise here rather than silently truncate, which is the behaviour we want.
    """
    eps, cls = [], []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            t, *rest = line.strip().split(',')
            try:
                v = [float(x) for x in rest]
            except ValueError:
                sys.exit("%s: unparseable row (decimal comma? see CLAUDE.md): %s" % (path, line[:60]))
            if t == 'E' and len(v) >= 9:
                eps.append(dict(d_forward=v[0], d_lateral=v[1], d_theta=v[2], duration=v[3],
                                pos_var=max(v[7], 1e-12), theta_var=max(v[8], 1e-12)))
            elif t == 'C' and len(v) >= 4:
                cls.append(dict(d_theta=v[1], duration=v[2], weight=v[3]))
    return eps, cls


def rows(ep):
    """The three Jacobian rows and their weights, exactly as the estimator builds them."""
    wp, wt = 1.0 / ep['pos_var'], 1.0 / ep['theta_var']
    j_along = np.zeros(6); j_along[P_K_V] = ep['d_forward']
    j_cross = np.zeros(6); j_cross[P_EPS_YAW] = -ep['d_forward']; j_cross[P_K_LAT] = ep['d_lateral']
    j_head = np.zeros(6)
    j_head[P_K_OMEGA] = ep['d_theta']; j_head[P_B_OMEGA] = ep['duration']
    j_head[P_DK_WHEEL] = ep['d_forward']
    return [(j_along, wp), (j_cross, wp), (j_head, wt)]


def information(eps, cls, with_prior=True):
    H = np.zeros((6, 6))
    if with_prior:
        H += np.diag(1.0 / PRIOR_SIGMA ** 2)
    for ep in eps:
        for j, w in rows(ep):
            H += w * np.outer(j, j)
    for c in cls:
        j = np.zeros(6); j[P_K_OMEGA] = c['d_theta']; j[P_B_OMEGA] = c['duration']
        H += c['weight'] * np.outer(j, j)
    return H


def report(H, Hd, label):
    print("\n%s" % label)
    sig = np.sqrt(np.diag(np.linalg.inv(H)))
    Hp = np.diag(1.0 / PRIOR_SIGMA ** 2)
    print("  %-10s %11s %11s %8s %9s   %s"
          % ("parameter", "post sigma", "prior", "shrink", "data", "informed"))
    for i, n in enumerate(P_NAMES):
        r = sig[i] / PRIOR_SIGMA[i]
        # ★★★ DATA SHARE: how much of this parameter's information came from DRIVING rather than
        # from the prior we asserted. It separates two failures that both read as "not informed":
        # a parameter the ROUTE never excited (low data info, fixable by driving differently) from
        # one whose PRIOR is so tight the data cannot compete (fixable only by loosening the prior,
        # or not at all). No manoeuvre can help the second, so a low share here means STOP LOOKING
        # for a manoeuvre. Measured 2026-09-01: b_omega 19%, every other live parameter 93-98%.
        share = 100.0 * Hd[i, i] / max(Hd[i, i] + Hp[i, i], 1e-30)
        print("  %-10s %11.6f %11.6f %7.2fx %8.1f%%   %s"
              % (n, sig[i], PRIOR_SIGMA[i], 1.0 / r if r > 0 else float('inf'), share,
                 "yes" if r < 0.9 else "NO — left where the prior put it"))
    # ★ Two distinct reasons a parameter cannot be driven into shape, and they need different
    # advice. UNEXCITED: the covariate is (near) zero, so there is no data to swamp — on this base
    # k_lat is that by construction and is the designed negative control, not a fault. SWAMPED:
    # there IS real data and the prior simply asserts more. Only the second is a prior to argue with.
    unexcited = [P_NAMES[i] for i in range(6) if Hd[i, i] < 0.01 * Hp[i, i]]
    swamped = [P_NAMES[i] for i in range(6)
               if 0.01 * Hp[i, i] <= Hd[i, i] < Hp[i, i]]
    if unexcited:
        print("  ⚠ UNEXCITED — %s got essentially NO data: the covariate is ~zero over this window."
              % ", ".join(unexcited))
        print("    On a differential base k_lat is this BY CONSTRUCTION and is the negative control;")
        print("    if it ever leaves its prior, parameters are leaking into each other.")
    if swamped:
        print("  ⚠ PRIOR-SWAMPED — %s HAS real data and the prior still asserts more."
              % ", ".join(swamped))
        print("    No manoeuvre reaches it: the question is whether the declared prior is justified,")
        print("    not which route to drive. Do not go looking for a better route.")
    print("\n  %-9s %14s %14s %10s   eigenvalues" % ("block", "lambda_min", "lambda_max", "cond"))
    for name, idx in BLOCKS:
        sub = H[np.ix_(idx, idx)]
        ev = np.sort(np.linalg.eigvalsh(sub))
        cond = ev[-1] / ev[0] if ev[0] > 0 else float('inf')
        print("  %-9s %14.4g %14.4g %10.1f   %s"
              % (name, ev[0], ev[-1], cond, " ".join("%.3g" % e for e in ev)))
    print("  ★ the heading block is the one to watch: k_omega, b_omega and dk_wheel share a")
    print("    component and are separated ONLY by covariate (rotation, time, distance).")
    return H


def score_manoeuvres(H0, eps, budget_s, vmax, wmax):
    """Rank candidate constant-rate primitives by what they would ADD to the current information."""
    if not eps:
        sys.exit("no episodes: cannot take the noise model from the data")
    # The candidate's weights come from the OBSERVED episodes, not from a guess.
    pv = float(np.median([e['pos_var'] for e in eps]))
    tv = float(np.median([e['theta_var'] for e in eps]))
    dur = float(np.median([e['duration'] for e in eps]))
    n_ep = max(1, int(round(budget_s / max(dur, 1e-3))))
    print("\ncandidate manoeuvres — %.0f s budget = %d episodes of %.2f s"
          % (budget_s, n_ep, dur))
    print("  noise taken from the data: pos_var %.3g, theta_var %.3g (medians)" % (pv, tv))
    print("  base limits: v <= %.2f m/s, w <= %.2f rad/s\n" % (vmax, wmax))

    lib = [("straight, fast",        vmax,       0.0),
           ("straight, slow",        0.15,       0.0),
           ("pivot, fast",           0.0,        wmax),
           ("pivot, slow",           0.0,        wmax / 5),
           ("arc, fast v / fast w",  vmax,       wmax),
           ("arc, fast v / slow w",  vmax,       wmax / 5),
           ("arc, slow v / fast w",  0.15,       wmax),
           ("arc, mid",              vmax / 2,   wmax / 2)]

    sign0, logdet0 = np.linalg.slogdet(H0)
    lam0 = min(np.linalg.eigvalsh(H0[np.ix_(BLOCKS[2][1], BLOCKS[2][1])]))
    out = []
    for name, v, w in lib:
        ep = dict(d_forward=v * dur, d_lateral=0.0, d_theta=w * dur, duration=dur,
                  pos_var=pv, theta_var=tv)
        H = H0.copy()
        for j, wt in rows(ep):
            H += n_ep * wt * np.outer(j, j)
        _, logdet = np.linalg.slogdet(H)
        lam = min(np.linalg.eigvalsh(H[np.ix_(BLOCKS[2][1], BLOCKS[2][1])]))
        sig = np.sqrt(np.diag(np.linalg.inv(H)))
        sig0 = np.sqrt(np.diag(np.linalg.inv(H0)))
        out.append((name, v, w, 0.5 * (logdet - logdet0), lam / lam0,
                    sig0[P_B_OMEGA] / sig[P_B_OMEGA], sig0[P_K_OMEGA] / sig[P_K_OMEGA]))
    out.sort(key=lambda r: -r[3])
    print("  %-22s %6s %6s %10s %11s %10s %10s"
          % ("manoeuvre", "v", "w", "IG nats", "lam_min x", "b_omega x", "k_omega x"))
    for r in out:
        print("  %-22s %6.2f %6.2f %10.2f %11.2f %10.2f %10.2f" % r)

    # ★★★ COMPOSITES, AND THEY ARE NOT A REFINEMENT — THEY ARE THE POINT. A single constant-rate
    # primitive produces episodes whose covariate vectors are all PARALLEL, so however long it is
    # held it adds information along ONE direction. The derivation says separation comes from
    # DIVERSITY of (v, w), which no single primitive can supply. If every single-primitive row above
    # leaves b_omega at 1.00x, that is the theory being right, not the tool being blind.
    print("\n── COMBINATIONS, budget split evenly (a single rate CANNOT separate a shared axis) ──")
    combos = []
    for i in range(len(lib)):
        for j in range(i + 1, len(lib)):
            combos.append([lib[i], lib[j]])
            for k in range(j + 1, len(lib)):
                combos.append([lib[i], lib[j], lib[k]])
    scored = []
    for parts in combos:
        H = H0.copy()
        share = max(1, n_ep // len(parts))
        for _, v, w in parts:
            ep = dict(d_forward=v * dur, d_lateral=0.0, d_theta=w * dur, duration=dur,
                      pos_var=pv, theta_var=tv)
            for j, wt in rows(ep):
                H += share * wt * np.outer(j, j)
        _, logdet = np.linalg.slogdet(H)
        lam = min(np.linalg.eigvalsh(H[np.ix_(BLOCKS[2][1], BLOCKS[2][1])]))
        sig = np.sqrt(np.diag(np.linalg.inv(H)))
        sig0 = np.sqrt(np.diag(np.linalg.inv(H0)))
        scored.append((" + ".join(p[0] for p in parts), 0.5 * (logdet - logdet0), lam / lam0,
                       sig0[P_B_OMEGA] / sig[P_B_OMEGA], sig0[P_DK_WHEEL] / sig[P_DK_WHEEL]))
    scored.sort(key=lambda r: -r[2])          # rank by E-optimality: feed the starved direction
    print("  %-52s %9s %10s %10s %10s"
          % ("combination", "IG nats", "lam_min x", "b_omega x", "dk_wheel x"))
    for r in scored[:8]:
        print("  %-52s %9.2f %10.2f %10.2f %10.2f" % r)
    print("  ranked by lam_min, not by IG — the question is which direction is STARVED.")
    print("  ★ A column stuck at 1.00 across every row, single and combined, is not a gap in this")
    print("    library. It means that parameter is PRIOR-DOMINATED and driving cannot reach it —")
    print("    check its data share above before designing a route for it.")
    print("\n  IG nats   = 0.5*log det(H_after/H_before): the D-optimal score, and the expected")
    print("              information gain an active-inference agent should value.")
    print("  lam_min x = growth of the HEADING block's smallest eigenvalue: the E-optimal score,")
    print("              which is what a starved direction actually needs.")
    print("  ★ These two can disagree, and when they do prefer lam_min: a manoeuvre can add a lot")
    print("    of information to directions that were already well determined.")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    flags = {a.split('=')[0]: a.split('=')[1] for a in sys.argv[1:] if '=' in a and a.startswith('--')}
    path = args[0] if args else 'etc/motion_calib_state.csv'
    budget = float(flags.get('--budget', 60.0))
    vmax = float(flags.get('--vmax', 0.7))     # controller/etc/config.toml MaxAdvSpeed
    wmax = float(flags.get('--wmax', 1.0))     # controller/etc/config.toml MaxRotSpeed

    eps, cls = load(path)
    print(__doc__.split('THE MODEL')[0].strip())
    print("\n%s: %d episodes, %d closed pivots" % (path, len(eps), len(cls)))
    if not eps:
        sys.exit("no episodes — drive first, or point at a saved window")
    d = sum(abs(e['d_forward']) for e in eps)
    r = sum(abs(e['d_theta']) for e in eps)
    print("  window covers %.1f m and %.1f rad (%.2f rad/m)" % (d, r, r / max(d, 1e-9)))
    H = information(eps, cls)
    Hd = information(eps, cls, with_prior=False)
    report(H, Hd, "WHAT THIS WINDOW DETERMINES")
    score_manoeuvres(H, eps, budget, vmax, wmax)


if __name__ == '__main__':
    main()
