#!/usr/bin/env python3
"""Graded-quantity ground truth read from the VISUAL meshes (what the sensors see), not from boundingObject.

    mesh_extents.py <wbt> <DEF-or-name>      e.g.  mesh_extents.py threerooms-piso.wbt BED_1

The spec (docs/superpowers/specs/2026-11-23-shared-engine-design.md, "Graded quantities") names ONE layer per class:

  class        graded layer (this file)                                 source of the vertices
  bed          mattress (w 1.571, length 1.971) and its box centre      Webots asset cache, Bed/meshes/mattress.obj, object 'mattress'
  nightstand   body box                                                 the world's own painted_wooden_nightstand.obj (whole mesh)
  toilet       footprint of bowl + tank                                 Toilet.proto IndexedFaceSets x the proto's Transform scale 0.2
  chair        seat centre + yaw                                        Chair.proto, DEF SEAT, the vertices BELOW the largest gap in z
  table        top centre, extent, yaw                                  the world's table1.dae, the vertices ABOVE the largest gap in z
  refrigerator body footprint centre                                    the world's fridge.dae (whole mesh)
  hood         canopy box                                               the world node's first IndexedFaceSet (frustum), NOT the chimney Box

"Largest gap in z" separates a slab from the legs / backrest by the mesh's own vertex heights, with no cut-off chosen.
A layer is a centred box (lx, ly, z0, z1, ox, oy) in the OBJECT frame: size along the object x and y, z range, and the
box centre (ox, oy) from the object's origin. Those six numbers are what measure_mask_error.py calls an extent.
The Webots protos come from the asset cache (~/.cache/Cyberbotics/Webots/assets, file name = sha1 of the raw URL), the
meshes of the world from the world's own folder. A class with no mesh layer here (Desk, WoodenTable, RoundTable) has
no entry: callers keep their boundingObject number for it and say so.
"""
import hashlib, math, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.join(HERE, '..', '..', 'room_concept', 'estimation_scripts', 'scripts')
CACHE = os.path.expanduser('~/.cache/Cyberbotics/Webots/assets')
RAW = 'https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/'   # the URLs the world's EXTERNPROTOs use


def asset(rel):
    """Local path of a Webots asset (proto or mesh) under projects/objects/, from the asset cache."""
    p = os.path.join(CACHE, hashlib.sha1((RAW + rel).encode()).hexdigest())
    if not os.path.exists(p):
        raise FileNotFoundError(f'Webots asset not in the cache ({p}); open the world once in Webots to cache {RAW + rel}')
    return p


# ---------------------------------------------------------------- vertices of the sources
def obj_vertices(path, obj_name=None):
    """Vertices of a Wavefront .obj; with obj_name only those under `o <obj_name>`."""
    out, cur = [], None
    for l in open(path):
        if l.startswith('o '): cur = l[2:].strip()
        elif l.startswith('v ') and (obj_name is None or cur == obj_name): out.append([float(t) for t in l.split()[1:4]])
    return np.array(out)


def dae_vertices(path):
    import xml.etree.ElementTree as ET
    v = [np.array(e.text.split(), float).reshape(-1, 3) for e in ET.parse(path).getroot().iter()
         if e.tag.endswith('float_array') and e.get('id', '').endswith('positions-array')]
    return np.vstack(v)


def file_vertices(path):
    return dae_vertices(path) if path.endswith('.dae') else obj_vertices(path)


def obj_extents(path):
    """Bounding box (lo, hi) of every vertex of a .obj or Collada .dae mesh, in the mesh's own frame."""
    v = file_vertices(path)
    return v.min(0), v.max(0)


def points_of(text):
    """Every `point [ ... ]` of a VRML text, as one (N, 3) array."""
    return np.vstack([np.array(m.split(), float).reshape(-1, 3) for m in re.findall(r'point \[([^\]]*)\]', text)])


def split_at_largest_z_gap(v, keep):
    """The vertices below ('low') or above ('high') the largest gap between consecutive vertex heights."""
    z = np.unique(np.round(v[:, 2], 4)); k = int(np.argmax(np.diff(z)))
    return v[v[:, 2] <= z[k]] if keep == 'low' else v[v[:, 2] > z[k]]


def box_of(v):
    """(lx, ly, z0, z1, ox, oy): the axis-aligned box of the vertices, centre (ox, oy) in the object frame."""
    lo, hi = v.min(0), v.max(0)
    return (float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(lo[2]), float(hi[2]), float((hi[0] + lo[0]) / 2), float((hi[1] + lo[1]) / 2))


def _first_ifs(node):
    """Points of the first IndexedFaceSet under a parsed node."""
    if isinstance(node, dict):
        if node.get('_type') == 'IndexedFaceSet':
            return np.array(node['coord']['point'], float).reshape(-1, 3)
        for v in node.values():
            r = _first_ifs(v)
            if r is not None: return r
    elif isinstance(node, list):
        for v in node:
            r = _first_ifs(v)
            if r is not None: return r
    return None


def _urls(n):
    if isinstance(n, dict):
        for k, v in n.items():
            if k == 'url': yield from _flat(v)
            else: yield from _urls(v)
    elif isinstance(n, list):
        for v in n: yield from _urls(v)


def _flat(v):
    if isinstance(v, str): yield v.strip('"')
    elif isinstance(v, list):
        for e in v: yield from _flat(e)


def _mesh_of(node, wbt):
    """Vertices of the world mesh (.obj / .dae) a node draws, in the object frame."""
    us = [u for u in _urls(node) if u.endswith(('.obj', '.dae'))]
    if not us: raise ValueError('node draws no .obj/.dae mesh')
    return file_vertices(os.path.join(os.path.dirname(wbt), us[0]))


# ---------------------------------------------------------------- the graded layers
def layer_vertices(cls, node=None, wbt=None):
    """Vertices (N, 3), object frame, of the graded layer of a class; None when the class has no mesh layer here."""
    if cls == 'bed':
        return obj_vertices(asset('bedroom/protos/Bed/meshes/mattress.obj'), 'mattress')
    if cls == 'toilet':
        t = open(asset('bathroom/protos/Toilet.proto')).read()
        i = t.index('Transform {'); s = float(re.match(r'Transform \{\s*scale ([-0-9.e]+)', t[i:]).group(1))
        return points_of(t[i:]) * s
    if cls == 'chair':
        t = open(asset('chairs/protos/Chair.proto')).read()
        i = t.index('DEF SEAT')
        seat = np.array(re.search(r'coord Coordinate \{\s*point \[([^\]]*)\]', t[i:]).group(1).split(), float).reshape(-1, 3)
        return split_at_largest_z_gap(seat, 'low')
    if cls == 'nightstand': return _mesh_of(node, wbt)
    if cls == 'refrigerator': return _mesh_of(node, wbt)
    if cls == 'table': return split_at_largest_z_gap(_mesh_of(node, wbt), 'high') if list(_urls(node)) else None
    if cls == 'hood': return _first_ifs(node)
    return None


def layer_box(cls, node=None, wbt=None):
    v = layer_vertices(cls, node, wbt)
    return None if v is None else box_of(v)


# Extents of the layers whose meshes are not in the world folder, for measure_mask_error.py and nees_persistent.py: one source.
# Toilet: the footprint of bowl + tank (+ lid), Toilet.proto x 0.2.  Bed: the mattress.
def _known():
    return {'toilet': layer_box('toilet'), 'bed': layer_box('bed')}


KNOWN_EXTENTS = _known()


# ---------------------------------------------------------------- the world
CLASS_OF_NAME = (('Fridge', 'refrigerator'), ('KitchenHood', 'hood'), ('Table1', 'table'), ('Table1(1)', 'table'))


def classify(node):
    ty, nm = node['_type'], _name(node)
    if ty == 'Chair': return 'chair'
    if ty == 'Bed': return 'bed'
    if ty == 'Toilet': return 'toilet'
    if nm.startswith('nightstand'): return 'nightstand'
    for k, c in CLASS_OF_NAME:
        if nm == k: return c
    return None


def _name(n):
    v = n.get('name')
    while isinstance(v, list) and v: v = v[0]
    return v.strip('"') if isinstance(v, str) else ''


def world_nodes(wbt):
    """[(def_name or '', name, class, node)] for every graded-class node (top level and inside Groups)."""
    sys.path.insert(0, SC)
    import furniture as F
    t = F.tokens(open(wbt).read()); out = []

    def visit(n, d):
        if n['_type'] == 'Group':
            for c in n.get('children', []):
                if isinstance(c, dict): visit(c, '')
            return
        c = classify(n)
        if c: out.append((d, _name(n) or n['_type'], c, n))
    i = 0
    while i < len(t):
        if t[i] == 'EXTERNPROTO': i += 2; continue
        if t[i] == 'DEF' and i + 3 < len(t) and t[i + 3] == '{': d = t[i + 1]; i += 2
        else: d = ''
        if i + 1 < len(t) and t[i + 1] == '{': n, i = F.parse_node(t, i); visit(n, d)
        else: i += 1
    return out


def def_node(wbt, d):
    """Parse the node written `DEF <d> <Type> {` (at any depth)."""
    sys.path.insert(0, SC)
    import furniture as F
    txt = open(wbt).read()
    m = re.search(r'DEF ' + re.escape(d) + r' (\w+) \{', txt)
    if not m: return None
    t = F.tokens(txt[m.start():])
    return F.parse_node(t, 2)[0]


def graded(wbt, key):
    """(name, class, x, y, z, yaw, node) of the object `key` (a DEF name or a Webots name)."""
    sys.path.insert(0, SC)
    import furniture as F
    n = def_node(wbt, key)
    if n is not None:
        c = classify(n) or ('table' if n['_type'] == 'Desk' else None)
        nm = _name(n) or n['_type']
    else:
        hit = [(nm, c, n) for _, nm, c, n in world_nodes(wbt) if nm == key]
        if not hit: raise KeyError(f'{key}: no DEF or graded-class node of that name in {wbt}')
        nm, c, n = hit[0]
    x, y, z, yaw = F.T(n)
    return nm, c, x, y, z, yaw, n


def to_world(x, y, yaw, ox, oy):
    return x + math.cos(yaw) * ox - math.sin(yaw) * oy, y + math.sin(yaw) * ox + math.cos(yaw) * oy


def world_layers(wbt):
    """{name: dict(cls, x, y, yaw, box, cx, cy)} for every graded-class object of the world with a mesh layer; x, y is the Webots
    origin, (cx, cy) the layer box centre in the world."""
    sys.path.insert(0, SC)
    import furniture as F
    out = {}
    for d, nm, c, n in world_nodes(wbt):
        try: b = layer_box(c, n, wbt)
        except (ValueError, KeyError): b = None
        if b is None: continue
        x, y, _, yaw = F.T(n)
        cx, cy = to_world(x, y, yaw, b[4], b[5])
        out[nm] = dict(cls=c, x=x, y=y, yaw=yaw, box=b, cx=cx, cy=cy)
    return out


def main(argv):
    wbt, key = argv[0], argv[1]
    nm, c, x, y, z, yaw, n = graded(wbt, key)
    v = layer_vertices(c, n, wbt)
    if v is None: print(f'{nm}: class {c} has no mesh layer here (kept as boundingObject)'); return 1
    b = box_of(v); cx, cy = to_world(x, y, yaw, b[4], b[5])
    print(f'{nm} class {c}: Webots origin ({x:.4f}, {y:.4f}) yaw {yaw:.4f} rad')
    print(f'  graded layer, object frame: size {b[0]:.4f} x {b[1]:.4f}, z {b[2]:.4f} .. {b[3]:.4f}, box centre ({b[4]:+.4f}, {b[5]:+.4f})')
    print(f'  graded layer, world: centre ({cx:.4f}, {cy:.4f}) yaw {yaw:.4f} rad')
    print(f'  origin vs layer: bbox centre offset ({b[4]:+.4f}, {b[5]:+.4f}) = {math.hypot(b[4], b[5]):.4f} m; '
          f'vertex mean offset ({v[:, 0].mean():+.4f}, {v[:, 1].mean():+.4f}) = {math.hypot(v[:, 0].mean(), v[:, 1].mean()):.4f} m ({len(v)} vertices)')
    if c == 'hood':
        print('  hood: the origin is the canopy centre iff the offset above is ~0 (the Webots node origin is the frustum base centre, z = 0 of the canopy)')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
