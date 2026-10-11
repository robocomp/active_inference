#!/usr/bin/env python3
"""Furniture footprints (world XY polygons) of a Webots world -> furniture.json.
Parses the .wbt (VRML-like) into a node tree; for every top-level object takes its boundingObject (Box / Cylinder
primitives under Poses, or a Collada Mesh projected to the floor as the union of its triangles) and applies the
object's pose. Desk/Chair PROTOs use their R2025a bounding sizes. Walls, floor, ceiling and wall-mounted flat items
are skipped. usage: furniture.py <world.wbt> <out.json>"""
import sys, re, json, math, os, xml.etree.ElementTree as ET
import numpy as np, shapely
from shapely.geometry import Polygon, Point, box, mapping
from shapely.ops import unary_union
from shapely import affinity

SKIP = {'Ceiling', 'Floor', 'Wall', 'Rodapies', 'Poster', 'Whiteboard', 'GlassBay', 'Door', 'Door(1)', 'CableTray'}
Z_MIN, Z_MAX = 0.03, 2.2          # floor-standing structure the LiDAR / robot can meet

def tokens(s):
    s = re.sub(r'#[^\n]*', '', s)
    return re.findall(r'"[^"]*"|[{}\[\]]|[^\s{}\[\]"]+', s)

def parse_node(t, i):
    """t[i] is the type (DEF handled by caller); returns (node dict, next index)."""
    typ = t[i]; assert t[i + 1] == '{', (t[i], t[i + 1]); i += 2; f = {'_type': typ}
    while t[i] != '}':
        if t[i] == 'hidden': i += 1
        key = t[i]; i += 1
        v, j = parse_value(t, i)
        if j == i and t[i] not in '{}[]': j = i + 1     # enum/identifier value (e.g. NULL): skip it
        f[key], i = v, j
    return f, i + 1

def parse_value(t, i):
    if t[i] == '[':
        i += 1; out = []
        while t[i] != ']':
            v, j = parse_value(t, i)
            i = j if j > i else i + 1                    # an unrecognised token (enum word): skip it, never stall
            out.append(v)
        return out, i + 1
    if t[i] == 'DEF': i += 2
    if i + 1 < len(t) and t[i + 1] == '{' and re.match(r'[A-Z]', t[i]): return parse_node(t, i)
    if t[i] == 'USE': return {'_type': 'USE'}, i + 2
    vals = []
    while i < len(t) and t[i] not in '{}[]' and re.match(r'^[-+0-9.eE]+$|^TRUE$|^FALSE$|^"', t[i]) and not (vals and t[i].startswith('"')):
        vals.append(t[i]); i += 1
        if vals[0].startswith('"'): break
    return vals, i

def num(f, k, d):
    v = f.get(k); return [float(x) for x in v] if isinstance(v, list) and v and not isinstance(v[0], dict) else d

def T(node):
    """2-D pose (x, y, yaw) of a node; rotation assumed about +-z (all furniture here)."""
    x, y, z = num(node, 'translation', [0, 0, 0]); ax = num(node, 'rotation', [0, 0, 1, 0])
    yaw = ax[3] * (1 if ax[2] >= 0 else -1) if abs(ax[2]) > 0.9 else 0.0
    return x, y, z, yaw

def place(g, x, y, yaw): return affinity.translate(affinity.rotate(g, yaw, origin=(0, 0), use_radians=True), x, y)

def dae_footprint(path):
    root = ET.parse(path).getroot(); ns = {'c': root.tag.split('}')[0][1:]}
    tris = []
    for mesh in root.iterfind('.//c:mesh', ns):
        src = {s.get('id'): np.array(s.find('c:float_array', ns).text.split(), float) for s in mesh.iterfind('c:source', ns)}
        vin = mesh.find('c:vertices', ns); pos = src[vin.find("c:input[@semantic='POSITION']", ns).get('source')[1:]].reshape(-1, 3)
        for prim in list(mesh.iterfind('c:triangles', ns)) + list(mesh.iterfind('c:polylist', ns)):
            ins = prim.findall('c:input', ns); stride = max(int(x.get('offset')) for x in ins) + 1
            vo = int([x for x in ins if x.get('semantic') == 'VERTEX'][0].get('offset'))
            p = np.array(prim.find('c:p', ns).text.split(), int).reshape(-1, stride)[:, vo]
            vc = np.array([int(v) for v in prim.find('c:vcount', ns).text.split()]) if prim.tag.endswith('polylist') else np.full(len(p) // 3, 3)
            starts = np.r_[0, np.cumsum(vc)[:-1]]
            for n in np.unique(vc):                      # fan-triangulate each polygon size at once
                st = starts[vc == n]
                for j in range(1, n - 1):
                    tris.append(pos[p[np.c_[st, st + j, st + j + 1]]])
    if not tris: return None
    tri = np.concatenate(tris)                           # (N, 3 verts, xyz)
    keep = (tri[:, :, 2].max(1) >= Z_MIN) & (tri[:, :, 2].min(1) <= Z_MAX)
    tri = tri[keep][:, :, :2]
    a = 0.5 * np.abs((tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1]) - (tri[:, 2, 0] - tri[:, 0, 0]) * (tri[:, 1, 1] - tri[:, 0, 1]))
    tri = tri[a > 1e-8]
    if not len(tri): return None
    g = shapely.union_all(shapely.polygons(np.concatenate([tri, tri[:, :1]], axis=1)), grid_size=0.005)
    return g.buffer(0.005).buffer(-0.005)

def prims(node, x0=0, y0=0, z0=0, yaw0=0):
    """Box / Cylinder footprints under a boundingObject (Pose / Group / Transform nesting)."""
    out = []
    if not isinstance(node, dict): return out
    typ = node['_type']
    if typ in ('Pose', 'Transform'):
        x, y, z, yaw = T(node); c, s = math.cos(yaw0), math.sin(yaw0)
        X, Y = x0 + c * x - s * y, y0 + s * x + c * y
        ch = node.get('children', [])
        for k in (ch if isinstance(ch, list) else [ch]): out += prims(k, X, Y, z0 + z, yaw0 + yaw)
    elif typ == 'Group':
        for k in node.get('children', []): out += prims(k, x0, y0, z0, yaw0)
    elif typ == 'Shape': out += prims(node.get('geometry'), x0, y0, z0, yaw0)
    elif typ == 'Box':
        sx, sy, sz = num(node, 'size', [0, 0, 0])
        if z0 + sz / 2 >= Z_MIN and z0 - sz / 2 <= Z_MAX: out.append(place(box(-sx / 2, -sy / 2, sx / 2, sy / 2), x0, y0, yaw0))
    elif typ == 'Cylinder':
        r = num(node, 'radius', [0.5])[0]; out.append(Point(x0, y0).buffer(r, 32))
    return out

PROTO = {'Desk': [(-0.35, -0.6, 0.35, 0.6)], 'Chair': [(-0.25, -0.25, 0.25, 0.25)]}   # R2025a bounding objects (top / seat)

def main():
    wbt, out = sys.argv[1], sys.argv[2]; base = os.path.dirname(wbt)
    t = tokens(open(wbt).read()); i = 0; top = []
    while i < len(t):
        if t[i] == 'EXTERNPROTO': i += 2; continue
        if t[i] == 'DEF': i += 2; continue
        if i + 1 < len(t) and t[i + 1] == '{': n, i = parse_node(t, i); top.append(n)
        else: i += 1
    objs = []
    def visit(n):
        if n['_type'] == 'Group':
            for c in n.get('children', []): visit(c)
            return
        nm = n.get('name') or [f'"{n["_type"]}"']; nm = nm[0] if isinstance(nm, list) else nm; name = (nm[0] if isinstance(nm, list) else nm).strip('"')
        if name in SKIP or n['_type'] in ('PointLight', 'Shadow', 'TexturedBackground', 'TexturedBackgroundLight', 'WorldInfo', 'Viewpoint', 'BeerBottle'): return
        x, y, z, yaw = T(n); geo = []
        if n['_type'] in PROTO: geo = [place(box(*b), x, y, yaw) for b in PROTO[n['_type']]]
        else:
            bo = n.get('boundingObject')
            if isinstance(bo, dict) and bo['_type'] == 'Mesh':
                u = bo['url'][0]; u = u[0] if isinstance(u, list) else u
                fp = dae_footprint(os.path.join(base, u.strip('"')))
                if fp is not None: geo = [place(fp, x, y, yaw)]
            else: geo = [place(g, x, y, yaw) for g in prims(bo)]
        if geo:
            g = unary_union(geo)
            if g.area > 0.005: objs.append(dict(name=name, type=n['_type'], area=round(g.area, 3), geometry=mapping(g)))
    for n in top: visit(n)
    json.dump(objs, open(out, 'w'), indent=1)
    for o in objs: print(f"{o['name']:18s} {o['type']:6s} {o['area']:.2f} m2")

if __name__ == '__main__': main()
