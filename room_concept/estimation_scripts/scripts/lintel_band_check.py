#!/usr/bin/env python3
"""lintel_band_check.py -- does a LiDAR height band ABOVE THE DOOR LINTEL keep through-door returns out (Lucas Bonilla's
TFM: "only the points where p_z is between z_min and z_max ... the higher points")?

3-D ray model of the helios on Shadow: 32 layers evenly over +54.5..-15.5 deg (inverted H32F70, ROBOT_GEOMETRY.md),
at 1.075 + 0.0325 m above the floor; 720 azimuths.  World = threerooms-piso.wbt walls (adoption_mock parser, full
height), DOOR_0 / DOOR_2 shut (wall), DOOR_1 open with a lintel at 2.1 m (its frameHeight), ceiling 3.0 m (room node
room_height 3.005), floor 0.  A return is a wall/lintel hit; ceiling and floor hits are not wall points.
For each band [lo, hi] (world z of the return) and robot position on DOOR_1's axis: in-band wall returns, and how many of
them lie BEYOND DOOR_1's plane (the other room) = rogue.
CAVEAT: layer spacing assumed uniform; furniture not modelled (the parser keeps walls only).
"""
import math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adoption_mock as am

Z0 = 1.075 + 0.0325
LAYERS = np.radians(np.linspace(-15.5, 54.5, 32))
CEIL, LINTEL = 3.0, 2.1


def hits_2d(segs, o, a):
    """ranges along azimuth a to every segment (sorted), with the segment index"""
    d = np.array([math.cos(a), math.sin(a)])
    out = []
    for i, (p, q) in enumerate(segs):
        e = q - p; den = d[0] * e[1] - d[1] * e[0]
        if abs(den) < 1e-12: continue
        w = p - o
        r = (w[0] * e[1] - w[1] * e[0]) / den
        u = (w[0] * d[1] - w[1] * d[0]) / den
        if r > 1e-4 and 0 <= u <= 1: out.append((r, i))
    return sorted(out)


def main():
    W = am.parse_world()
    d1 = W.doors["DOOR_1"]
    segs = list(W.segments); kinds = ["wall"] * len(segs)
    for name, d in W.doors.items():
        c, t = d["centre"], d["tangent"]
        segs.append((c - d["jamb_in"] * t, c + d["jamb_in"] * t)); kinds.append("lintel" if name == "DOOR_1" else "wall")
    segs = [(np.asarray(p, float), np.asarray(q, float)) for p, q in segs]
    c1, n1 = d1["centre"], d1["normal"]
    bands = [(2.03, 3.0, "current (2.0 body .. top)"), (2.15, 2.9, "above lintel 2.15..2.9"), (2.2, 2.8, "above lintel 2.2..2.8")]
    for side, label in ((-1, "in room_2"), (+1, "in room_1")):
        for dist in (3.0, 2.0, 1.0, 0.5, 0.3):
            o = c1 + side * dist * n1 if True else None
            # which side of DOOR_1 is "the other room" for this position
            res = {b: [0, 0] for b in bands}
            for az in np.linspace(0, 2 * math.pi, 720, endpoint=False):
                H = hits_2d(segs, o, az)
                for e in LAYERS:
                    te = math.tan(e)
                    r_ceil = (CEIL - Z0) / te if te > 1e-9 else math.inf
                    r_floor = -Z0 / te if te < -1e-9 else math.inf
                    end = None; under = False
                    for r, i in H:
                        if r >= min(r_ceil, r_floor): break
                        z = Z0 + r * te
                        if kinds[i] == "wall" or (kinds[i] == "lintel" and z > LINTEL):
                            end = (r, z); break
                        if kinds[i] == "lintel": under = True     # passed through DOOR_1's opening, under the lintel
                    if end is None: continue
                    r, z = end
                    pt = o + r * np.array([math.cos(az), math.sin(az)])
                    beyond = under
                    for b in bands:
                        if b[0] <= z <= b[1]:
                            res[b][0] += 1; res[b][1] += int(beyond)
            line = "  ".join(f"{b[2]}: {res[b][1]}/{res[b][0]} rogue ({100.0 * res[b][1] / max(1, res[b][0]):.1f}%)" for b in bands)
            print(f"{label} (y={o[1]:.2f}) {dist:.1f} m from DOOR_1 | {line}")


if __name__ == "__main__":
    main()
