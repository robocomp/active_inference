#!/usr/bin/env python3
"""Responsibility table for spec lidar-measures 7c.3 (rev 3.1), computed from the FORMULA, not from code.

  p(r_i)       = m * N(r; t, s_r^2 + s_box^2) + (1 - m) * p(r | rest)
  p(r | rest)  = (1 - w_u) * N(r; t_rest, s_r^2 + s_rest^2) + w_u / L     (a rest surface is predicted inside the box)
               = 1 / L                                                    (none predicted)
  responsibility r = m * N_box / p(r)

Constants (spec): s_r^2 = 1/2500 + 0.02^2 (LidarPrecision + Helios sigma_sensor), s_rest = 0.05, L = 0.6 m, w_u = 0.2.
s_box is the box component's sigma ALONG THE RAY (J S_L J^T, in metres). All distances in cm in the table, metres inside.
Output uses '.' decimals through explicit format specs (no locale).
"""
import math

SR2 = 1.0 / 2500.0 + 0.02 ** 2
S_REST = 0.05
L = 0.6
W_U = 0.2


def npdf(x, var):
    return math.exp(-0.5 * x * x / var) / math.sqrt(2.0 * math.pi * var)


def resp(m, e_box, s_box, e_rest=None, s_rest=S_REST):
    """e_box: return minus the belief face's predicted range; e_rest: return minus the rest surface's range (None = none)."""
    nb = npdf(e_box, SR2 + s_box ** 2)
    if e_rest is None:
        prest = 1.0 / L
    else:
        prest = (1.0 - W_U) * npdf(e_rest, SR2 + s_rest ** 2) + W_U / L
    return m * nb / (m * nb + (1.0 - m) * prest)


def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


SB = (0.0, 0.02, 0.04, 0.06)


def row(label, vals):
    print("| %-46s | %s |" % (label, " | ".join("%5.3f" % v for v in vals)))


def main():
    print("Constants: sigma_r^2 = 1/2500 + 0.02^2 = %.5f m^2 (sigma_r = %.1f cm), sigma_rest = 5 cm, L = 0.6 m, w_u = 0.2" % (SR2, 100 * math.sqrt(SR2)))
    print()
    print("A. TRUE return on the object (m = 0.9); belief face offset by e, box sigma along the ray 5 cm")
    print("| rest prediction                                | e = 10 cm | e = 14 cm | e = 23 cm |")
    print("|---|---|---|---|")
    for lab, er in (("room hit in chord (wall 30 cm behind the return)", 0.30), ("none predicted", None)):
        print("| %-46s | %s |" % (lab, " | ".join("%9.3f" % resp(0.9, e, 0.05, er) for e in (0.10, 0.14, 0.23))))
    print()
    hdr = "| case (m = 0.1 unless stated)                   | s_box 0 | s_box 2 | s_box 4 | s_box 6 |"
    print("B/C. FOREIGN returns; the belief face is e in front of the return; s_box in cm")
    print(hdr); print("|---|---|---|---|---|")
    for g in (0.02, 0.05):
        print("| wall %d cm behind the face (rest = the wall)    | %s |" % (round(100 * g), " | ".join("%7.3f" % resp(0.1, g, s, 0.0) for s in SB)))
    for g in (0.02, 0.05):
        print("| neighbour %d cm off, UNPUBLISHED (rest = none)  | %s |" % (round(100 * g), " | ".join("%7.3f" % resp(0.1, g, s, None) for s in SB)))
    print()
    print("D. COINCIDENT foreign box (return at e = 0 on this object and on the foreign box; s_box = s_rest = 5 cm)")
    print("| m = 0.9 | m = 0.5 |"); print("|---|---|")
    print("| %7.3f | %7.3f |" % (resp(0.9, 0.0, 0.05, 0.0), resp(0.5, 0.0, 0.05, 0.0)))
    print()
    m = 0.9 * phi(3.0)
    print("E. OVER-SEGMENTED wall return: pi = 0.9 at s = +3 sigma_px gives m = %.4f; the return is on the wall (rest = the wall)" % m)
    print(hdr.replace("case (m = 0.1 unless stated)    ", "belief face e in front of the wall           ")); print("|---|---|---|---|---|")
    for g in (0.02, 0.05):
        print("| e = %d cm                                       | %s |" % (round(100 * g), " | ".join("%7.3f" % resp(m, g, s, 0.0) for s in SB)))


if __name__ == "__main__":
    main()
