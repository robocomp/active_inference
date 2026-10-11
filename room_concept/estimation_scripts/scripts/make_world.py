#!/usr/bin/env python3
"""10 random Manhattan rooms (port of wall_slam_selftest.cpp random_room) in ONE Webots world, 30 m apart.
Writes the world, rooms.json (GT polygons in world coords + start poses) and starts.csv for run_campaign.sh."""
import json, math, random, sys, numpy as np
from shapely.geometry import Polygon, box, Point
from shapely.ops import unary_union
OUT = sys.argv[1]; BASE = sys.argv[2]   # output dir, piso.wbt to take the header/robot from
T_WALL, H_WALL, SPACING = 0.12, 3.0, 30.0

def random_room(seed, order):
    rng = random.Random(seed); U = rng.random
    grow = math.sqrt(max(1, order) / 3.0)
    W = (4.5 + 3.5 * U()) * max(1.0, grow); H = (3.5 + 2.5 * U()) * max(1.0, grow)
    shell = box(-W / 2, -H / 2, W / 2, H / 2); poly = shell; feats = []
    for k in range(1, order):
        for _ in range(40):
            face = int(U() * 4) % 4; carve = U() < 0.6
            along = (0.4 + 1.1 * U()) if carve else (0.9 + 1.1 * U())   # no thin outward fins
            deep = (0.3 + 0.8 * U()) if carve else (0.5 + 0.6 * U())
            t = 0.12 + 0.76 * U()
            if face in (0, 2):
                y0 = -H / 2 + t * (H - along); x = -W / 2 if face == 0 else W / 2
                s = 1 if face == 0 else -1          # inward direction
                b = box(*sorted([x, x + s * deep]) [:1], y0, *sorted([x, x + s * deep])[1:], y0 + along) if carve else \
                    box(*sorted([x, x - s * deep])[:1], y0, *sorted([x, x - s * deep])[1:], y0 + along)
            else:
                x0 = -W / 2 + t * (W - along); y = -H / 2 if face == 1 else H / 2
                s = 1 if face == 1 else -1
                ys = sorted([y, y + s * deep]) if carve else sorted([y, y - s * deep])
                b = box(x0, ys[0], x0 + along, ys[1])
            if any(b.distance(f) < 0.6 for f in feats): continue      # features must not touch / merge
            cand = poly.difference(b) if carve else poly.union(b)
            if cand.geom_type != 'Polygon' or len(cand.interiors) or not cand.is_valid: continue
            if cand.buffer(-0.55).geom_type != 'Polygon' or cand.buffer(-0.55).is_empty: continue  # stays one drivable region
            poly = cand; feats.append(b); break
    poly = poly.simplify(1e-6)
    return Polygon(np.round(np.array(poly.exterior.coords)[:-1], 4)).normalize() if False else poly

def l_room(seed):
    """L: a W x H rectangle minus one corner block; both arms at least 1.8 m wide (drivable, explorer must turn the corner)."""
    rng = random.Random(seed); U = rng.random
    W, H = 6.0 + 3.0 * U(), 5.0 + 3.0 * U()
    cw, ch = min(W - 1.8, W * (0.35 + 0.25 * U())), min(H - 1.8, H * (0.35 + 0.25 * U()))
    sx, sy = rng.choice([-1, 1]), rng.choice([-1, 1])              # which corner is cut
    cx = W / 2 - cw if sx > 0 else -W / 2; cy = H / 2 - ch if sy > 0 else -H / 2
    return box(-W / 2, -H / 2, W / 2, H / 2).difference(box(cx, cy, cx + cw, cy + ch)).simplify(1e-6)

def u_room(seed):
    """U: a W x H rectangle with a notch cut from one side's middle, deep enough that the two arms hide each other's
    far ends from most of the base; arms and base at least 1.8 m wide. Random orientation (multiple of 90 deg)."""
    from shapely.affinity import rotate
    rng = random.Random(seed); U = rng.random
    W, H = 6.5 + 2.5 * U(), 5.5 + 2.0 * U()
    nw = min(W - 3.6, W * (0.30 + 0.15 * U()))                     # notch width: both arms >= 1.8 m
    nd = min(H - 1.8, H * (0.55 + 0.15 * U()))                     # notch depth: base >= 1.8 m
    P = box(-W / 2, -H / 2, W / 2, H / 2).difference(box(-nw / 2, H / 2 - nd, nw / 2, H / 2 + 0.01)).simplify(1e-6)
    P = Polygon(np.round(np.array(P.exterior.coords)[:-1], 4))
    return rotate(P, 90 * rng.randrange(4), origin=(0, 0)).simplify(1e-6)

def t_room(seed):
    """T: a bar plus a perpendicular stem off its middle third; random orientation (multiple of 90 deg)."""
    from shapely.affinity import rotate
    rng = random.Random(seed); U = rng.random
    bw, bh = 7.0 + 3.0 * U(), 2.2 + 1.0 * U()
    sw, sh = 2.2 + 1.0 * U(), 3.0 + 2.0 * U()
    sx = (-0.15 + 0.3 * U()) * bw                                   # stem centre along the bar
    P = unary_union([box(-bw / 2, 0, bw / 2, bh), box(sx - sw / 2, -sh, sx + sw / 2, 0.0001)]).simplify(1e-6)
    P = Polygon(np.round(np.array(P.exterior.coords)[:-1], 4))
    return rotate(P, 90 * rng.randrange(4), origin=(0, 0)).simplify(1e-6)

def walls(P):
    V = np.array(P.exterior.coords)[:-1]
    if Polygon(V).exterior.is_ccw is False: V = V[::-1]
    n = len(V); out = []
    for i in range(n):
        a, b = V[i], V[(i + 1) % n]; pa, pn = V[i - 1], V[(i + 2) % n]
        conv = lambda p, q, r: (q[0]-p[0])*(r[1]-q[1]) - (q[1]-p[1])*(r[0]-q[0]) > 0
        d = (b - a) / np.linalg.norm(b - a); nout = np.array([d[1], -d[0]])   # CCW: outward = right of travel
        a2 = a - d * (T_WALL if conv(pa, a, b) else 0); b2 = b + d * (T_WALL if conv(a, b, pn) else 0)
        c = (a2 + b2) / 2 + nout * T_WALL / 2; L = np.linalg.norm(b2 - a2)
        out.append((c, L, math.atan2(d[1], d[0])))
    return out

def solid(name, c, z, size, yaw=0.0, color='0.85 0.85 0.8'):
    return f'''    Solid {{
      translation {c[0]:.4f} {c[1]:.4f} {z:.4f}
      rotation 0 0 1 {yaw:.6f}
      children [ Shape {{ appearance PBRAppearance {{ baseColor {color} roughness 0.9 metalness 0 }} geometry Box {{ size {size[0]:.4f} {size[1]:.4f} {size[2]:.4f} }} }} ]
      name "{name}"
      boundingObject Box {{ size {size[0]:.4f} {size[1]:.4f} {size[2]:.4f} }}
      locked TRUE
    }}
'''

FURNISH = '--furniture' in sys.argv     # 2026-10-11: furnished variant of the same 10 rooms
ROBOT_R = 0.55                           # half the narrowest passage the CONTROLLER drives: body + margin (0.43 let a
                                         # table leave 0.85 m gaps that sealed room 8's stem, 2026-10-11)

def furnish(P, rng):
    """Axis-aligned furniture for room P (rectilinear, CCW): worktops (0.9 m) and tall cabinets (2.0 m) against the
    walls, tables (top on four legs) in the open. Each item is kept only if the room minus all items, eroded by the
    robot radius, is still one region. Returns [dict(name, kind, box=(x0,y0,x1,y1), parts=[(cx,cy,cz,sx,sy,sz)])]."""
    U = rng.random; items = []
    V = np.array(P.exterior.coords)[:-1]
    if not Polygon(V).exterior.is_ccw: V = V[::-1]
    def ok(b, gap):
        if not P.buffer(1e-6).contains(b): return False
        if any(b.distance(box(*it['box'])) < gap for it in items): return False
        free = P.difference(unary_union([box(*it['box']) for it in items] + [b]))
        er = free.buffer(-ROBOT_R)
        return er.geom_type == 'Polygon' and not er.is_empty and er.area > 0.5 * P.buffer(-ROBOT_R).area
    n_wall = 2 + int(P.area // 25); tries = 0
    while sum(it['kind'] != 'table' for it in items) < n_wall and tries < 400:
        tries += 1
        k = rng.randrange(len(V)); a, b = V[k], V[(k + 1) % len(V)]; L = np.linalg.norm(b - a)
        tall = U() < 0.4; depth = 0.4 if tall else 0.6; h = 2.0 if tall else 0.9
        ln = min(L - 0.3, (0.8 + 0.8 * U()) if tall else (1.2 + 1.2 * U()))
        if ln < 0.6: continue
        d = (b - a) / L; nin = np.array([-d[1], d[0]])               # CCW: interior is left of travel
        s0 = 0.15 + U() * (L - 0.3 - ln); p0 = a + d * s0 + nin * 0.01; p1 = p0 + d * ln + nin * depth
        bx = box(min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]), max(p0[1], p1[1]))
        if not ok(bx, 0.9): continue
        x0, y0, x1, y1 = bx.bounds
        items.append(dict(name=('cabinet' if tall else 'worktop') + str(len(items)), kind='cabinet' if tall else 'worktop', box=bx.bounds,
                          parts=[((x0 + x1) / 2, (y0 + y1) / 2, h / 2, x1 - x0, y1 - y0, h)]))
    n_tab = 1 if P.area < 45 else 2; tries = 0
    while sum(it['kind'] == 'table' for it in items) < n_tab and tries < 400:
        tries += 1
        w, l = (0.8, 1.2) if U() < 0.5 else (1.2, 0.8)
        x0b, y0b, x1b, y1b = P.bounds; cx, cy = rng.uniform(x0b, x1b), rng.uniform(y0b, y1b)
        bx = box(cx - w / 2, cy - l / 2, cx + w / 2, cy + l / 2)
        if bx.distance(P.exterior) < 0.9 or not ok(bx, 0.9): continue
        legs = [(cx + sx * (w / 2 - 0.05), cy + sy * (l / 2 - 0.05), 0.355, 0.05, 0.05, 0.71) for sx in (-1, 1) for sy in (-1, 1)]
        items.append(dict(name='table' + str(len(items)), kind='table', box=bx.bounds, parts=[(cx, cy, 0.73, w, l, 0.04)] + legs))
    return items

orders = [2, 3, 4, 5, 6, 7, 3, 4, 5, 6]
kinds  = ['random', 'L', 'random', 'T', 'random', 'random', 'L', 'random', 'T', 'random']   # 2026-10-11: rooms 1,6 L; 3,8 T
if '--kinds' in sys.argv:            # e.g. --kinds U,U,U  (a separate world; seeds 1000+i as usual)
    kinds = sys.argv[sys.argv.index('--kinds') + 1].split(',')
    orders = [0] * len(kinds)
rooms, body = [], []
for i, order in enumerate(orders):
    kind = kinds[i]
    P = random_room(1000 + i, order) if kind == 'random' else l_room(1000 + i) if kind == 'L' else u_room(1000 + i) if kind == 'U' else t_room(1000 + i)
    P = Polygon(np.round(np.array(P.exterior.coords)[:-1], 6))
    off = np.array([-SPACING * i, 0.0])
    from shapely.affinity import translate
    Pw = translate(P, *off)
    for k, (c, L, yaw) in enumerate(walls(Pw)):
        body.append(solid(f'room{i}_wall{k}', c, H_WALL / 2, (L, T_WALL, H_WALL), yaw))
    x0, y0, x1, y1 = Pw.bounds
    body.append(solid(f'room{i}_ceiling', ((x0 + x1) / 2, (y0 + y1) / 2), H_WALL + 0.025, (x1 - x0 + 1, y1 - y0 + 1, 0.05), 0, '0.95 0.95 0.95'))
    body.append(f'''    PointLight {{ attenuation 0 0 1 intensity 3.2 location {(x0+x1)/2:.3f} {(y0+y1)/2:.3f} 2.8 radius 18 }}\n''')
    items = furnish(Pw, random.Random(500 + i)) if FURNISH else []
    color = {'worktop': '0.55 0.4 0.3', 'cabinet': '0.35 0.35 0.45', 'table': '0.6 0.45 0.25'}
    for it in items:
        for q, (cx, cy, cz, sx, sy, sz) in enumerate(it['parts']):
            body.append(solid(f"room{i}_{it['name']}_{q}", (cx, cy), cz, (sx, sy, sz), 0, color[it['kind']]))
    rng = random.Random(7 + i); inner = Pw.buffer(-0.8)          # start: >= 0.8 m from every wall, random heading
    if items: inner = inner.difference(unary_union([box(*it['box']) for it in items]).buffer(0.8))
    while True:
        p = Point(rng.uniform(x0, x1), rng.uniform(y0, y1))
        if inner.contains(p): break
    rooms.append(dict(id=i, kind=kind, order=order if kind == 'random' else 0, seed=1000 + i, offset=off.tolist(), verts=len(P.exterior.coords) - 1,
                      area=round(P.area, 3), polygon=[list(map(float, v)) for v in np.array(Pw.exterior.coords)[:-1]],
                      start=[round(p.x, 3), round(p.y, 3), round(rng.uniform(-math.pi, math.pi), 3)],
                      furniture=[dict(name=it['name'], kind=it['kind'], box=list(it['box']), height=max(z + h / 2 for _, _, z, _, _, h in it['parts'])) for it in items]))
xmin = -SPACING * (len(orders) - 1) - 15
floor = solid('synthetic_floor', ((xmin + 15) / 2, 0), -0.025, (15 - xmin, 30, 0.05), 0, '0.6 0.55 0.5')
base = open(BASE).read().split('\n')
head = '\n'.join(base[:193])                                    # EXTERNPROTOs, WorldInfo (wheel contact), Viewpoint, lights
robot = f'''DEF shadow Shadow {{
  translation {rooms[0]["start"][0]} {rooms[0]["start"][1]} 0.0323
  rotation 0 0 1 {rooms[0]["start"][2]}
}}'''
open(f'{OUT}/synthetic_rooms.wbt', 'w').write(head + '\nGroup {\n  children [\n' + floor + ''.join(body) + '  ]\n}\n' + robot + '\n')
json.dump(rooms, open(f'{OUT}/rooms.json', 'w'), indent=1)
with open(f'{OUT}/starts.csv', 'w') as f:
    f.write('id,x,y,theta\n')
    for r in rooms: f.write(f"{r['id']},{r['start'][0]},{r['start'][1]},{r['start'][2]}\n")
if FURNISH:   # footprints for the layout mosaic (regrade_anchor.py FURNITURE=...)
    from shapely.geometry import mapping
    json.dump([dict(name=f"room{r['id']}_{f['name']}", type=f['kind'], area=round(box(*f['box']).area, 3), geometry=mapping(box(*f['box'])))
               for r in rooms for f in r['furniture']], open(f'{OUT}/furniture.json', 'w'), indent=1)
for r in rooms: print(r['id'], r['kind'], 'furniture', len(r['furniture']), 'order', r['order'], 'verts', r['verts'], 'area', r['area'], 'start', r['start'])
