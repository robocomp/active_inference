#!/usr/bin/env python3
"""Offline generator for rc::RobotFootprint::shadow() — Shadow's TRUE simulated body.

body  = XY outline of the vertices of shadow.stl (binary, metres, z-up, x lateral, y forward = robot frame) with z <= 1.45
wheels= the TWO drive wheels of Shadow.proto: Cylinder r 0.1, h 0.05 along the lateral axle, centred x = +-0.259, y = 0
        => rectangles x in [0.234, 0.284] (and mirrored), y in [-0.10, 0.10]
Hulls the union, simplifies (CONSERVATIVE: replaces edges by the intersection of their neighbours' lines, so the result contains the raw hull; growth <= +0.5 %), prints the
C++ vertex list CCW. Usage: shadow_footprint.py shadow.stl [out_stem]   (out_stem given => also writes out_stem.stl and out_stem.obj: the body+wheels mesh)
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

def rot_axis_angle(ax, ang):
    ax = np.asarray(ax, float); ax = ax / np.linalg.norm(ax); c, s_ = np.cos(ang), np.sin(ang)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + s_ * K + (1 - c) * (K @ K)

def xf(tris, R, t):
    return [tuple(R @ v + t for v in tr) for tr in tris]

def cylinder_z(r, half, seg=32):
    """Webots Cylinder: axis = local Z (Shadow.proto's own comment), closed caps, centred at the origin."""
    ring = [(r*np.cos(2*np.pi*k/seg), r*np.sin(2*np.pi*k/seg)) for k in range(seg + 1)]
    tris = []
    for k in range(seg):
        (x0, y0), (x1, y1) = ring[k], ring[k+1]
        lo0, lo1, hi0, hi1 = (x0, y0, -half), (x1, y1, -half), (x0, y0, half), (x1, y1, half)
        tris += [(lo0, lo1, hi1), (lo0, hi1, hi0), ((0, 0, half), hi0, hi1), ((0, 0, -half), lo1, lo0)]
    return [tuple(np.array(v, float) for v in t) for t in tris]

def sphere(r, nlat=12, nlon=24):
    tris = []
    P = lambda i, j: np.array([r*np.sin(np.pi*i/nlat)*np.cos(2*np.pi*j/nlon), r*np.sin(np.pi*i/nlat)*np.sin(2*np.pi*j/nlon), r*np.cos(np.pi*i/nlat)])
    for i in range(nlat):
        for j in range(nlon):
            a, b, c, d = P(i, j), P(i, j+1), P(i+1, j+1), P(i+1, j)
            if i > 0: tris.append((a, d, b) if False else (a, b, d))
            if i < nlat - 1: tris.append((b, c, d))
    return tris

def box(sx, sy, sz):
    h = np.array([sx, sy, sz]) / 2
    c = [np.array([x*h[0], y*h[1], z*h[2]]) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    q = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return [t for a, b, c_, d in q for t in ((c[a], c[b], c[c_]), (c[a], c[c_], c[d]))]

def shadow_parts():
    """Every Shape / boundingObject geometry Shadow.proto attaches to the robot, in the SAME frame as shadow.stl (the proto's body
    Shape has no offset; Robot translation 0 0 0.033 applies to everything alike). name -> triangles. The cylinder-Z convention and the
    transforms are those of Shadow.proto (line numbers in comments)."""
    Rx90 = rot_axis_angle((1, 0, 0), -1.5708)
    parts = {}
    # drive wheels (344-372, 407-): Solid translation (+-0.259,0,0.067) rot z 1.5708, Pose rot x -1.5708 -> Cylinder r 0.1 h 0.05, axis = lateral x
    for nm, sx in (("wheel_left", -1), ("wheel_right", +1)):
        parts[nm] = xf(xf(cylinder_z(0.10, 0.025), Rx90, 0), rot_axis_angle((0, 0, 1), 1.5708), np.array([sx*0.259, 0, 0.067]))
    # casters (466-, 506-): Sphere r 0.04 at (0, +-0.18, 0.007)
    parts["caster_front"] = xf(sphere(0.04), np.eye(3), np.array([0, 0.18, 0.007]))
    parts["caster_back"] = xf(sphere(0.04), np.eye(3), np.array([0, -0.18, 0.007]))
    # THETA 360 camera lens bumps (40-75): camera at (0,-0.17,1.275) rot z -+pi/2; lens spheres r 0.0105 at local x +-0.0115
    for nm, ang in (("camera_360_2_lens", -1.5707953), ("camera_360_1_lens", +1.5707953)):
        R = rot_axis_angle((0, 0, 1), ang if nm.endswith("2_lens") else -ang)
        for sgn in (+1, -1):
            parts[f"{nm}{'+' if sgn > 0 else '-'}"] = xf(sphere(0.0105, 8, 16), np.eye(3), R @ np.array([sgn*0.0115, 0, 0]) + np.array([0, -0.17, 1.275]))
    # helios (83-138): Lidar at (0,-0.155,1.075) rot (1,1,0,pi); Cylinder h 0.10 r 0.055 at local (-0.0125,0,0)
    Rh = rot_axis_angle((1, 1, 0), 3.14159265359)
    parts["helios_housing"] = xf(cylinder_z(0.055, 0.05), Rh, Rh @ np.array([-0.0125, 0, 0]) + np.array([0, -0.155, 1.075]))
    # bpearl (206-238): Lidar at (0,0.14,0.67); Sphere r 0.05
    parts["bpearl_dome"] = xf(sphere(0.05), np.eye(3), np.array([0, 0.14, 0.67]))
    # boundingObject Group (540-): Box 0.44x0.46x0.76 at z 0.38 and the mast Box 0.1x0.1x0.61 at (0,-0.17,1.03)
    parts["bounding_body"] = xf(box(0.44, 0.46, 0.76), np.eye(3), np.array([0, 0, 0.38]))
    parts["bounding_mast"] = xf(box(0.1, 0.1, 0.61), np.eye(3), np.array([0, -0.17, 1.03]))
    return parts

def keep(tris, zlo=0.0, zhi=1.8):
    z = np.array([v[2] for t in tris for v in t])
    return z.max() >= zlo and z.min() <= zhi

def write_meshes(tri_body, out_stem):
    tris = [tuple(np.asarray(v, float) for v in t) for t in tri_body]
    for nm, pt in shadow_parts().items():
        if keep(pt): tris += pt
    # outward orientation check on the wheel side faces is by construction; STL
    with open(out_stem + '.stl', 'wb') as f:
        f.write(b'shadow_footprint: shadow.stl + two drive wheels (tools/shadow_footprint.py)'.ljust(80, b' '))
        f.write(np.uint32(len(tris)).tobytes())
        for a, b, c in tris:
            n = np.cross(b - a, c - a); nn = np.linalg.norm(n); n = n / nn if nn > 0 else n
            f.write(np.array([*n, *a, *b, *c], '<f4').tobytes() + b'\0\0')
    idx, verts, faces = {}, [], []
    for t in tris:
        face = []
        for v in t:
            key = tuple(np.round(v, 6))
            if key not in idx: idx[key] = len(verts) + 1; verts.append(key)
            face.append(idx[key])
        if len(set(face)) == 3: faces.append(face)
    with open(out_stem + '.obj', 'w') as f:
        f.write('# shadow_footprint: shadow.stl + two drive wheels (r 0.10, w 0.05, x=+-0.259, z=0.067); generated by shadow_footprint.py\n')
        for v in verts: f.write('v %.6f %.6f %.6f\n' % v)
        for a, b, c in faces: f.write('f %d %d %d\n' % (a, b, c))
    print(f"// wrote {out_stem}.stl/.obj: {len(tris)} triangles, {len(verts)} vertices")

def main():
    raw = open(sys.argv[1], 'rb').read()
    n = int(np.frombuffer(raw, np.uint32, 1, 80)[0])
    rec = np.frombuffer(raw, np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]), n, 84)
    v = rec['v'].reshape(-1, 3).astype(float)
    if len(sys.argv) > 2: write_meshes(rec['v'].astype(float), sys.argv[2])
    print(f"// {n} triangles, bbox min {v.min(0)} max {v.max(0)}")
    v = v[(v[:, 2] <= 1.8) & (v[:, 2] >= 0.0)]
    body = hull(v[:, :2])
    print("// parts (frame of shadow.stl = robot frame), z range kept [0.0, 1.8]:")
    print(f"//   shadow.stl            x[{body[:,0].min():+.4f},{body[:,0].max():+.4f}] y[{body[:,1].min():+.4f},{body[:,1].max():+.4f}]")
    pv = [body]
    for nm, pt in shadow_parts().items():
        a_ = np.array([vv for t in pt for vv in t])
        print(f"//   {nm:<22} x[{a_[:,0].min():+.4f},{a_[:,0].max():+.4f}] y[{a_[:,1].min():+.4f},{a_[:,1].max():+.4f}] z[{a_[:,2].min():+.3f},{a_[:,2].max():+.3f}] {'KEPT' if keep(pt) else 'dropped'}")
        if keep(pt): pv.append(a_[:, :2])
    full = hull(np.vstack(pv))
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
