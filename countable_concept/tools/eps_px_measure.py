#!/usr/bin/env python3
"""Measure eps_px, the probability that the ZED mask label of a LiDAR return is wrong (spec 2026-11-15-lidar-measures
§7c.8 (a0) "The label is never certain"; plan Task 15b).

usage: python3 -I eps_px_measure.py RUN_DIR [--scene SCENE_CSV] [--room NAME=x,y,rot]

RUN_DIR holds <cls>_label_dump.csv (the agents' label_dump.csv, written with Model.LabelDumpEnabled; the controller copies
them here with the class prefix like *_ai2_log.csv), <cls>_ai2_log.csv (to associate each instance with its GT object by
nearest final centre, as lidar_measures_score does), gt_pose.tsv / gt_error.csv / proto_room.csv (the per-run room->world
fit, imported from lidar_measures_score) and the scene csv (default tools/scene_threerooms.csv, the scorer's).

Per return (endpoint in the ROOM frame -> world through the fitted room frame): ON the object iff within 2 cm of ANY of that
instance's ground-truth PART boxes (bed_*; ns_body + its ns_leg; toilet_*), else foreign. Per class:
  eps_out = P(in_image, s/sigma < -3 | on object)      the label says "not this" on the object
  eps_in  = P(in_image, s/sigma > +3 | foreign)        the label says "this" off the object
  eps_px  = max(eps_out, eps_in)
plus the share of returns with in_image = 0, and b*(eps_px): the root b > 0 of  eps/(1-eps) * St(b)/p_pass + G(-b) = 1,
St the normalised Student-t with nu = p0 c^2 - 1, scale^2 = c^2/nu (p0 = 2500, c = 0.05), G its CDF (so G(-b) = 1/2 - int_0^b St),
p_pass = 1/6. Numbers are read with float(); no locale.
"""
import argparse, csv, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lidar_measures_score as S   # noqa: E402  (loaders: fit_room_frames, room_tf, load_instances, fnum, ROOMS, ROOM_OF)

ON_TOL = 0.02
NSIG = 3.0
P0, C, P_PASS = 2500.0, 0.05, 1.0 / 6.0


def load_parts(path):
    """-> {cls: {name: [box, ...]}} with box = (xmin, xmax, ymin, ymax, zmin, zmax); the scorer's grouping, but PARTS kept."""
    rows = list(csv.DictReader(open(path)))
    box = lambda r: tuple(S.fnum(r[k]) for k in ('xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'))
    parts = [(r['label'], box(r)) for r in rows]
    out = {'bed': {}, 'nightstand': {}, 'toilet': {}}
    bed = [b for l, b in parts if l.startswith('bed_')]
    if bed: out['bed']['bed'] = bed
    legs = [b for l, b in parts if l == 'ns_leg']
    for i, nb in enumerate(sorted([b for l, b in parts if l == 'ns_body'], key=lambda b: b[0])):
        mine = [g for g in legs if nb[0] - 0.05 <= g[0] and g[1] <= nb[1] + 0.05 and nb[2] - 0.05 <= g[2] <= nb[3] + 0.05]
        out['nightstand']['nightstand_%d' % i] = [nb] + mine
    toi = [b for l, b in parts if l.startswith('toilet_')]
    if toi: out['toilet']['toilet'] = toi
    return out


def centre(boxes):
    return (0.5 * (min(b[0] for b in boxes) + max(b[1] for b in boxes)), 0.5 * (min(b[2] for b in boxes) + max(b[3] for b in boxes)))


def near_box(p, b, tol):
    return all(b[2 * k] - tol <= p[k] <= b[2 * k + 1] + tol for k in range(3))


def bstar(eps):
    """Root b > 0 of eps/(1-eps)*St(b)/p_pass + G(-b) = 1; None if the left side is already <= 1 at b = 0 (no b > 0)."""
    nu = P0 * C * C - 1.0
    sc = C / math.sqrt(nu)
    lg = math.lgamma(0.5 * (nu + 1)) - math.lgamma(0.5 * nu) - 0.5 * math.log(nu * math.pi)
    St = lambda b: math.exp(lg - 0.5 * (nu + 1) * math.log1p((b / sc) ** 2 / nu)) / sc
    def G_neg(b, n=2000):   # G(-b) = 1/2 - int_0^b St (Simpson)
        if b <= 0.0: return 0.5
        h = b / n
        acc = St(0.0) + St(b) + sum((4 if k % 2 else 2) * St(k * h) for k in range(1, n))
        return 0.5 - acc * h / 3.0
    f = lambda b: eps / (1 - eps) * St(b) / P_PASS + G_neg(b) - 1.0
    if not 0 < eps < 1 or f(0.0) <= 0: return None
    lo, hi = 0.0, 1.0
    while f(hi) > 0 and hi < 1e3: hi *= 2
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run')
    ap.add_argument('--scene', default=S.SCENE)
    ap.add_argument('--room', action='append', default=[], metavar='NAME=x,y,rot', help='override a room origin')
    a = ap.parse_args()
    parts = load_parts(a.scene)
    fits = S.fit_room_frames(a.run)
    rooms = {}
    for k in S.ROOMS:
        f = fits[k]
        if 'origin' in f and 'error' not in f: rooms[k] = (f['origin'][0], f['origin'][1], f['rot'])
    for s in a.room:
        k, v = s.split('=')
        rooms[k] = tuple(float(x) for x in v.split(','))
    print('eps_px_measure  (spec 7c.8 a0, plan Task 15b)   run %s' % a.run)
    print('on-object = within %.0f cm of any GT part box; label wrong = in_image and |s/sigma| > %.0f on the wrong side' % (100 * ON_TOL, NSIG))
    for room in S.ROOMS:
        print('room %s: %s' % (room, ('origin (%.3f, %.3f) rot %+.4f' % rooms[room]) if room in rooms else 'NO frame fit (use --room)'))
    for cls in S.CLASSES:
        path = os.path.join(a.run, cls + '_label_dump.csv')
        if not os.path.exists(path): print('%s: no %s_label_dump.csv' % (cls, cls)); continue
        room = S.ROOM_OF[cls]
        if room not in rooms: print('%s: room %s has no frame; skipped' % (cls, room)); continue
        to_world, _, _ = S.room_tf(room, rooms)
        inst, _ = S.load_instances(a.run, cls)
        rows = {}
        for r in csv.DictReader(open(path)):
            rows.setdefault(r['node'], []).append(r)
        n = {'on': 0, 'off': 0, 'on_img': 0, 'off_img': 0, 'out': 0, 'in': 0}
        for node, rr in sorted(rows.items()):
            last = inst.get(node, [None])[-1] if inst.get(node) else None
            if last is not None: cx, cy = to_world(last['cx'], last['cy'])
            else:
                xs = sorted(S.fnum(r['ex']) for r in rr); ys = sorted(S.fnum(r['ey']) for r in rr)
                cx, cy = to_world(xs[len(xs) // 2], ys[len(ys) // 2])
            name = min(parts[cls], key=lambda k: math.hypot(centre(parts[cls][k])[0] - cx, centre(parts[cls][k])[1] - cy))
            boxes = parts[cls][name]
            for r in rr:
                wx, wy = to_world(S.fnum(r['ex']), S.fnum(r['ey']))
                p = (wx, wy, S.fnum(r['ez']))
                on = any(near_box(p, b, ON_TOL) for b in boxes)
                key = 'on' if on else 'off'
                n[key] += 1
                if S.fnum(r['in_image']) == 1:
                    n[key + '_img'] += 1
                    z = S.fnum(r['s_px']) / S.fnum(r['sigma_px'])
                    if on and z < -NSIG: n['out'] += 1
                    if not on and z > NSIG: n['in'] += 1
            print('  %s -> GT %s (%d returns)' % (node, name, len(rr)))
        tot = n['on'] + n['off']
        e_out = n['out'] / n['on_img'] if n['on_img'] else float('nan')
        e_in = n['in'] / n['off_img'] if n['off_img'] else float('nan')
        vals = [v for v in (e_out, e_in) if v == v]
        eps = max(vals) if vals else float('nan')
        b = bstar(eps) if eps == eps else None
        print('%s: returns %d (on %d, foreign %d); in_image on %d foreign %d; not-in-image share %.3f' % (
            cls, tot, n['on'], n['off'], n['on_img'], n['off_img'], 1 - (n['on_img'] + n['off_img']) / tot if tot else float('nan')))
        print('%s: eps_out %d/%d = %s   eps_in %d/%d = %s   eps_px = %s   b*(eps_px) = %s' % (
            cls, n['out'], n['on_img'], S.fmt(e_out, '.4f'), n['in'], n['off_img'], S.fmt(e_in, '.4f'), S.fmt(eps, '.4f'),
            'n/a' if eps != eps else ('none (neutral at b=0)' if b is None else '%.3f' % b)))


if __name__ == '__main__':
    main()
