#!/usr/bin/env python3
"""Offline generator for rc::RobotFootprint::shadow() — Shadow's TRUE simulated body.

body  = XY outline of the vertices of shadow.stl (binary, metres, z-up, x lateral, y forward = robot frame) with z <= 1.45
wheels= the TWO drive wheels of Shadow.proto: Cylinder r 0.1, h 0.05 along the lateral axle, centred x = +-0.259, y = 0
        => rectangles x in [0.234, 0.284] (and mirrored), y in [-0.10, 0.10]
Hulls the union, simplifies (CONSERVATIVE: replaces edges by the intersection of their neighbours' lines, so the result contains the raw hull; growth <= +0.5 %), prints the
C++ vertex list CCW. Usage: shadow_footprint.py shadow.stl
"""
import sys, numpy as np

def hull(pts):
    pts = sorted(set(map(tuple, pts)))
    def half(ps):
        h = []
        for p in ps:
            while len(h) >= 2 and (h[-1][0]-h[-2][0])*(p[1]-h[-2][1]) - (h[-1][1]-h[-2][1])*(p[0]-h[-2][0]) <= 0: h.pop()
            h.append(p)
        return h
    lo, up = half(pts), half(pts[::-1])
    return np.array(lo[:-1] + up[:-1])

def area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * np.sum(x*np.roll(y, -1) - np.roll(x, -1)*y)

def main():
    raw = open(sys.argv[1], 'rb').read()
    n = int(np.frombuffer(raw, np.uint32, 1, 80)[0])
    rec = np.frombuffer(raw, np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]), n, 84)
    v = rec['v'].reshape(-1, 3).astype(float)
    print(f"// {n} triangles, bbox min {v.min(0)} max {v.max(0)}")
    v = v[v[:, 2] <= 1.45]
    body = hull(v[:, :2])
    wheels = [(s*x, y) for s in (1, -1) for x in (0.234, 0.284) for y in (-0.10, 0.10)]
    full = hull(np.vstack([body, wheels]))
    a0 = area(full)
    if area(full) < 0: full = full[::-1]
    p = full.copy()
    # CONSERVATIVE simplification: the result must CONTAIN the raw hull (an under-reported body is the unsafe direction for every
    # contact check). Replace edge (i,i+1) by the intersection of its two neighbouring edges' lines: that vertex lies OUTSIDE the
    # old polygon, so the polygon only GROWS. Greedy on least growth, total growth <= +0.5 % of the raw area.
    def cross(u, v): return u[0]*v[1] - u[1]*v[0]
    def drop(p, i):
        m = len(p); A, B, C, D = p[(i-1) % m], p[i], p[(i+1) % m], p[(i+2) % m]
        d1, d2 = B - A, C - D            # line A->B forward, line D->C backward
        den = cross(d1, d2)
        if abs(den) < 1e-12: return None
        t = cross(D - A, d2) / den
        X = A + t * d1
        if t < 1.0 or cross(X - A, X - D) == 0: return None   # intersection must lie beyond B (outward)
        if cross(C - D, X - D) < -1e-12 and False: return None
        idx = [j for j in range(m) if j not in (i, (i+1) % m)]
        q = [p[j] for j in range(m)]
        out = []
        for j in range(m):
            if j == i: out.append(X)
            elif j == (i+1) % m: continue
            else: out.append(q[j])
        return np.array(out)
    while len(p) > 3:
        best = None
        for i in range(len(p)):
            q = drop(p, i)
            if q is None or area(q) <= 0: continue
            g = area(q) - area(p)
            if g >= -1e-12 and (best is None or g < best[0]): best = (g, q)
        if best is None or area(best[1]) - a0 > 0.005 * a0: break
        p = best[1]
    print("// raw hull (test fixture, every vertex must be contained):")
    print("//   " + " ".join(f"{{{x:+.6f}f, {y:+.6f}f}}," for x, y in full))
    # CCW starting at the lowest-left vertex
    if area(p) < 0: p = p[::-1]
    print(f"// raw hull {len(full)} verts area {a0:.4f}; simplified {len(p)} verts area {area(p):.4f} ({100*(area(p)/a0-1):+.2f} %)")
    print(f"// x_max {p[:,0].max():.4f}  y_max {p[:,1].max():.4f}  y_min {p[:,1].min():.4f}")
    print("f.poly_ = {")
    for i in range(0, len(p), 4):
        print("    " + " ".join(f"{{{x:+.4f}f, {y:+.4f}f}}," for x, y in p[i:i+4]))
    print("};")
main()
