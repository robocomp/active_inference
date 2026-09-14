#!/usr/bin/env python3
"""Corner-level disagreement between the three layouts of the Webots `piso` apartment.

The paper's §V-E claim is about a CORNER innovation, so the quantity that matters is not
the wall offset but how far each MODEL CORNER sits from the corner the LiDAR actually
sees.  This matches corners between layouts (nearest-neighbour under the best rigid
alignment, one-to-one via the Hungarian assignment) and reports the distribution.
"""
import numpy as np, os
from shapely.geometry import Polygon, Point
from scipy.optimize import linear_sum_assignment, minimize

S = "/tmp/claude-1000/-home-pbustos-robocomp-components-active-inference-room-concept/c1e48398-243f-4bda-b915-064cb4be9944/scratchpad"
E = np.load(S + "/exact.npy")
A = np.load(S + "/agent.npy")
T = np.load(S + "/traced.npy")          # already rigidly aligned into the webots frame


def interior_angle(p, i):
    n = len(p)
    a, b, c = p[(i - 1) % n], p[i], p[(i + 1) % n]
    u, v = a - b, c - b
    ang = np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))
    # sign of the cross product says convex/reflex for a CCW polygon
    return ang if np.cross(b - a, c - b) > 0 else 360 - ang


def match(ref, other, tag, maxd=1.2):
    D = np.linalg.norm(ref[:, None, :] - other[None, :, :], axis=2)
    C = D.copy(); C[C > maxd] = 1e3
    r, c = linear_sum_assignment(C)
    keep = [(i, j) for i, j in zip(r, c) if D[i, j] <= maxd]
    d = np.array([D[i, j] for i, j in keep])
    print(f"\n=== corner offsets: {tag} ===")
    print(f"  {len(keep)}/{len(ref)} exact corners matched within {maxd} m "
          f"({len(other)} corners in the other layout)")
    print(f"  mean {d.mean():.4f}  median {np.median(d):.4f}  p90 {np.percentile(d,90):.4f}"
          f"  max {d.max():.4f}  RMS {np.sqrt((d**2).mean()):.4f}   [m]")
    print(f"  {'exact corner':>20} {'int.ang':>8} {'matched to':>20} {'offset':>8}")
    for i, j in sorted(keep, key=lambda k: -D[k[0], k[1]]):
        print(f"  ({ref[i][0]:8.3f},{ref[i][1]:7.3f}) {interior_angle(ref,i):7.1f}  "
              f"({other[j][0]:8.3f},{other[j][1]:7.3f})  {D[i,j]:7.4f}")
    unmatched = [k for k in range(len(ref)) if k not in [i for i, _ in keep]]
    if unmatched:
        print("  UNMATCHED exact corners: " + ", ".join(f"({ref[k][0]:.3f},{ref[k][1]:.3f})" for k in unmatched))
    return d


print("exact  %d corners, area %.4f, bbox %.3f x %.3f" % (len(E), Polygon(E).area, *np.ptp(E,0)))
print("agent  %d corners, area %.4f, bbox %.3f x %.3f" % (len(A), Polygon(A).area, *np.ptp(A,0)))
print("traced %d corners, area %.4f, bbox %.3f x %.3f (after alignment)" % (len(T), Polygon(T).area, *np.ptp(T,0)))

match(E, A, "EXACT mesh -> AGENT layout (apartamento_layout.svg)")
match(E, T, "EXACT mesh -> TRACED truth.txt (best rigid alignment)")

# --- the traced polygon under a PURE 180 deg (axis-preserving) alignment, translation only
Traw = None
line = open("datasets/webots_piso/truth.txt").read().strip().split(";")[1]
Traw = np.array([[float(v) for v in t.split(",")] for t in line.split()])
print("\nRAW truth.txt bbox: %.3f x %.3f m   vs exact %.3f x %.3f m  -> short by %.3f / %.3f m"
      % (*np.ptp(Traw,0), *np.ptp(E,0), np.ptp(E,0)[0]-np.ptp(Traw,0)[0], np.ptp(E,0)[1]-np.ptp(Traw,0)[1]))

R180 = Traw @ np.array([[-1.,0.],[0.,-1.]])
PE = Polygon(E)
def cost(t): return float(sum(PE.exterior.distance(Point(*(p+t)))**2 for p in R180))
res = minimize(cost, E.mean(0)-R180.mean(0), method="Nelder-Mead",
               options=dict(xatol=1e-7,fatol=1e-12,maxiter=20000,maxfev=20000))
T180 = R180 + res.x
d180 = match(E, T180, "EXACT mesh -> TRACED truth.txt (pure 180 deg, NO free rotation)")
