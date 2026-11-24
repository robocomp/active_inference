#!/usr/bin/env python3
"""Step 0, Step 4: how far is a ZED mask from the silhouette of the graded box?

    measure_mask_error.py <dump_dir> <wbt> [<slam_shadow.csv>] [--classes nightstand,bed,toilet]
                          [--extents bed=1.571,1.971,0,0.6 ...] [--hfov 1.9198] [--out frames.csv]

<dump_dir> is a retina [ZedDump] directory recorded with dump_masks = true (poses.csv, <stamp>.masks.csv,
<stamp>.masks.pix.csv; layout in retina/src/mask_dump_csv.h on feat/mask-dump).

For every dumped ZED mask of a graded class:
  1. the graded box (a centred box, extents lx x ly x [z0, z1] in the object frame, at the Webots pose of the
     matching object) is projected through the ZED pinhole at the frame's GROUND-TRUTH camera pose;
  2. the contour of the mask is sampled (every contour pixel) and each sample gets the SIGNED distance to the
     projected silhouette, positive when the mask edge lies OUTSIDE the silhouette (the mask is too big);
  3. per mask: mean offset (the part shared by the whole contour: a boundary that is grown or shrunk) and the
     std about it (the independent part).
Over frames:  sigma_px_boundary = std over frames of the per-frame mean offset;
              eps_px            = mean fraction of mask pixels outside the projected silhouette.

Conventions (each one is a measured fact or is derived, none is tuned):
  * ZED: pinhole, x right, y depth, z up, mounted at body (0, -0.075, +0.945) + body z 0.0325 (ROBOT_GEOMETRY.md; the
    dump's cam_z is 0.9775), level. Image 1280 x 720, hfov 1.9198 rad (Shadow.proto), so fx = fy = (W/2)/tan(hfov/2).
  * Ground truth: poses.csv gt_x, gt_y, gt_angle (Webots world; gt_angle = -(yaw + pi/2), place_stage.cpp), valid when
    gt_valid = 1. The camera sits at the robot position + (-0.075) along the heading. <slam_shadow.csv> is a MAP-frame
    estimate (no world transform in it), so it is accepted for the call signature but cannot supply a ground-truth pose.
  * Pixel (i, j) has its centre at (i, j). A mask contour passes through the centres of the boundary pixels, which lie
    on average half a pixel INSIDE the true edge (a boundary pixel centre is uniform in [0, 1) px inside it), so every
    offset gets +0.5 px.
  * A frame whose silhouette has a corner at or behind the camera is skipped (no finite silhouette). Contour samples
    within 2 px of the image border are dropped (the border cuts the mask, it is not a mask edge).
  * Occlusion by other objects is NOT modelled: the visible mask is smaller than the silhouette there. Report it with
    the numbers.
"""
import argparse, csv, math, os, re, sys
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'gt_mesh'))
from mesh_extents import KNOWN_EXTENTS, obj_extents   # THE source of the graded-layer extents (tools/gt_mesh)
SC = os.path.join(HERE, '..', '..', 'room_concept', 'estimation_scripts', 'scripts')
CAM_BACK_M = 0.075        # ZED sits 0.075 m behind the body origin (ROBOT_GEOMETRY.md)
HALF_PX = 0.5             # see the pixel-centre convention above
BORDER_PX = 2


# ---------------------------------------------------------------- geometry (pure, tested)
def intrinsics(w, h, hfov):
    f = (w / 2.0) / math.tan(hfov / 2.0)
    return f, f, w / 2.0, h / 2.0


def camera_pose(gt_x, gt_y, gt_angle, cam_z):
    """Camera (x, y, z, heading) in the Webots world from the dumped ground truth."""
    h = -gt_angle
    return gt_x - CAM_BACK_M * math.cos(h), gt_y - CAM_BACK_M * math.sin(h), cam_z, h


def box_corners(cx, cy, yaw, lx, ly, z0, z1, ox=0.0, oy=0.0):
    """8 corners of the box whose centre is (ox, oy) in the object frame (origin at (cx, cy), heading yaw)."""
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for sx in (-0.5, 0.5):
        for sy in (-0.5, 0.5):
            for z in (z0, z1):
                lx_, ly_ = ox + sx * lx, oy + sy * ly
                out.append((cx + c * lx_ - s * ly_, cy + s * lx_ + c * ly_, z))
    return np.array(out)


def project(points, cam, K):
    """world (N,3) -> pixel (N,2); None if any point is at or behind the image plane."""
    fx, fy, cx, cy = K
    x, y, z, h = cam
    d = points - np.array([x, y, z])
    f = np.array([math.cos(h), math.sin(h)]); r = np.array([math.sin(h), -math.cos(h)])
    xc, yc, zc = d[:, :2] @ r, d[:, :2] @ f, d[:, 2]
    if np.any(yc <= 1e-6):
        return None
    return np.stack([cx + fx * xc / yc, cy - fy * zc / yc], 1)


def silhouette(corners_px):
    return cv2.convexHull(corners_px.astype(np.float32))


def signed_offsets(samples, hull):
    """Signed distance (px) of each sample to the polygon boundary, positive OUTSIDE it."""
    return np.array([-cv2.pointPolygonTest(hull, (float(u), float(v)), True) for u, v in samples])


def mask_raster(pix, stride, h, w):
    m = np.zeros((h, w), np.uint8)
    for c, r in pix:
        m[int(r):int(r) + stride, int(c):int(c) + stride] = 1
    return m


def contour_samples(mask, border=BORDER_PX):
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return np.zeros((0, 2))
    p = np.vstack([c.reshape(-1, 2) for c in cs]).astype(float)
    h, w = mask.shape
    keep = (p[:, 0] >= border) & (p[:, 0] <= w - 1 - border) & (p[:, 1] >= border) & (p[:, 1] <= h - 1 - border)
    return p[keep]


def frame_stats(offsets):
    """(mean = shared part, std about it = independent part) of the contour offsets, half-pixel corrected."""
    o = np.asarray(offsets) + HALF_PX
    return float(o.mean()), float(o.std(ddof=1)) if len(o) > 1 else float('nan')


def fraction_outside(mask, hull):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return float('nan')
    out = sum(cv2.pointPolygonTest(hull, (float(x), float(y)), False) < 0 for x, y in zip(xs, ys))
    return out / len(xs)


# ---------------------------------------------------------------- world: graded boxes
def graded_boxes(wbt, classes, extents):
    """[(name, class, x, y, yaw, lx, ly, z0, z1)]; extents: class -> (lx, ly, z0, z1) from --extents, else from the
    object's own .obj mesh (bounding box in the object frame, centred in x, y)."""
    sys.path.insert(0, SC)
    import furniture as F
    t = F.tokens(open(wbt).read()); i = 0; top = []
    while i < len(t):
        if t[i] == 'EXTERNPROTO': i += 2; continue
        if t[i] == 'DEF': i += 2; continue
        if i + 1 < len(t) and t[i + 1] == '{': n, i = F.parse_node(t, i); top.append(n)
        else: i += 1
    out = []

    def name(n):
        v = n.get('name')
        while isinstance(v, list) and v: v = v[0]
        return v.strip('"') if isinstance(v, str) else ''

    def flat(v):
        if isinstance(v, str): yield v.strip('"')
        elif isinstance(v, list):
            for e in v: yield from flat(e)

    def urls(n):
        if isinstance(n, dict):
            for k, v in n.items():
                if k == 'url':
                    yield from flat(v)
                else: yield from urls(v)
        elif isinstance(n, list):
            for v in n: yield from urls(v)

    def cls(n):
        ty, nm = n['_type'], name(n)
        if ty == 'Bed': return 'bed'
        if nm == 'Fridge': return 'refrigerator'
        if nm in ('Table1', 'Table1(1)'): return 'table'
        if nm.startswith('nightstand'): return 'nightstand'
        if ty == 'Toilet': return 'toilet'
        return None

    def visit(n):
        if n['_type'] == 'Group':
            for c in n.get('children', []):
                if isinstance(c, dict): visit(c)
            return
        c = cls(n)
        if c not in classes: return
        x, y, _, yaw = F.T(n)
        if c in extents: lx, ly, z0, z1, ox, oy = (tuple(extents[c]) + (0.0, 0.0))[:6]
        else:
            objs = [u for u in urls(n) if u.endswith(('.obj', '.dae'))]
            if not objs: return
            lo, hi = obj_extents(os.path.join(os.path.dirname(wbt), objs[0]))
            lx, ly, z0, z1 = float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(lo[2]), float(hi[2])
            ox, oy = float(hi[0] + lo[0]) / 2, float(hi[1] + lo[1]) / 2
        out.append((name(n) or n['_type'], c, x, y, yaw, lx, ly, z0, z1, ox, oy))
    for n in top: visit(n)
    return out


# ---------------------------------------------------------------- the dump
def frames(dump_dir):
    with open(os.path.join(dump_dir, 'poses.csv'), newline='') as f:
        for r in csv.DictReader(f):
            if int(r['gt_valid']) != 1: continue
            s = int(r['stamp_ms'])
            if not os.path.exists(os.path.join(dump_dir, f'{s}.masks.csv')): continue
            yield s, float(r['gt_x']), float(r['gt_y']), float(r['gt_angle']), float(r['cam_z'])


def read_masks(dump_dir, stamp):
    with open(os.path.join(dump_dir, f'{stamp}.masks.csv'), newline='') as f:
        idx = list(csv.DictReader(f))
    pix = {}
    p = os.path.join(dump_dir, f'{stamp}.masks.pix.csv')
    if os.path.exists(p):
        with open(p, newline='') as f:
            rd = csv.reader(f); next(rd, None)
            for row in rd:
                if row: pix.setdefault(int(row[0]), []).append((float(row[1]), float(row[2])))
    return idx, pix


def measure(dump_dir, wbt, classes, extents, hfov):
    boxes = graded_boxes(wbt, classes, extents)
    rows = []
    for stamp, gx, gy, ga, cz in frames(dump_dir):
        idx, pix = read_masks(dump_dir, stamp)
        cam = camera_pose(gx, gy, ga, cz)
        for m in idx:
            if m['label'] not in classes or int(m['mask']) not in pix: continue
            w, h, stride = int(m['img_w']), int(m['img_h']), int(m['pixel_stride'])
            K = intrinsics(w, h, hfov)
            mk = mask_raster(pix[int(m['mask'])], stride, h, w)
            best = None
            for b in boxes:
                if b[1] != m['label']: continue
                px = project(box_corners(*b[2:5], *b[5:11]), cam, K)
                if px is None: continue
                hull = silhouette(px)
                sil = np.zeros((h, w), np.uint8); cv2.fillConvexPoly(sil, np.round(hull).astype(np.int32).reshape(-1, 2), 1)
                iou = float((sil & mk).sum()) / max(float((sil | mk).sum()), 1.0)
                if iou > 0 and (best is None or iou > best[0]): best = (iou, b, hull)
            if best is None: continue
            iou, b, hull = best
            ctr = np.array([[b[2] + math.cos(b[4]) * b[9] - math.sin(b[4]) * b[10], b[3] + math.sin(b[4]) * b[9] + math.cos(b[4]) * b[10], 0.5 * (b[7] + b[8])]])
            rng_m = float((ctr[0, :2] - np.array(cam[:2])) @ np.array([math.cos(cam[3]), math.sin(cam[3])]))   # depth along the optical axis
            cs = contour_samples(mk)
            if len(cs) < 2: continue
            mean, std = frame_stats(signed_offsets(cs, hull))
            rows.append(dict(stamp_ms=stamp, mask=int(m['mask']), cls=m['label'], gt_object=b[0], iou=iou,
                             range_m=rng_m, n_contour=len(cs), mean_offset_px=mean, resid_std_px=std,
                             frac_outside=fraction_outside(mk, hull)))
    return rows


def summarise(rows):
    out = {}
    for c in sorted({r['cls'] for r in rows}):
        R = [r for r in rows if r['cls'] == c]
        # one number per FRAME: the mean over the frame's masks of this class, so a frame with 3 masks counts once
        by = {}
        for r in R: by.setdefault(r['stamp_ms'], []).append(r)
        fm = [np.mean([r['mean_offset_px'] for r in v]) for v in by.values()]
        fo = [np.mean([r['frac_outside'] for r in v]) for v in by.values()]
        out[c] = dict(n_frames=len(by), n_masks=len(R),
                      sigma_px_boundary=float(np.std(fm, ddof=1)) if len(fm) > 1 else float('nan'),
                      mean_of_frame_means_px=float(np.mean(fm)), eps_px=float(np.mean(fo)),
                      median_resid_std_px=float(np.median([r['resid_std_px'] for r in R])))
    return out


def block_length(x):
    """Length of the blocks for a moving-block bootstrap of the series x: the first lag at which the autocorrelation
    falls inside the white-noise band +-2/sqrt(n) (beyond it, two frames that far apart are not distinguishable from
    independent), at least 1. Returns (block length, acf at lag 1)."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x); den = float(x @ x)
    if den == 0 or n < 8: return 1, float('nan')
    acf = [float(x[:n - k] @ x[k:]) / den for k in range(1, n // 2)]
    band = 2.0 / math.sqrt(n)
    for k, r in enumerate(acf, 1):
        if abs(r) < band: return k, acf[0]
    return n // 2, acf[0]


def ols_slope(z, y):
    z = np.asarray(z, float); y = np.asarray(y, float)
    zc = z - z.mean()
    return float(zc @ (y - y.mean()) / (zc @ zc))


def block_bootstrap_slope(z, y, L, n_boot=2000, seed=0):
    """Slope of y on z and its 95% interval from a moving-block bootstrap over the time-ordered frames (blocks of L)."""
    n = len(z); rg = np.random.default_rng(seed)
    nb = int(math.ceil(n / L)); out = []
    for _ in range(n_boot):
        ix = np.concatenate([np.arange(st, st + L) for st in rg.integers(0, n - L + 1, nb)])[:n]
        if np.ptp(z[ix]) == 0: continue
        out.append(ols_slope(z[ix], y[ix]))
    return ols_slope(z, y), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def range_bins(rows, f, n_bins=4, n_boot=2000, seed=0):
    """Is the per-frame mean offset constant in PIXELS or in METRES?  One value per frame (stamp) and class, frames in
    time order. Bins by range z: n_bins equal-COUNT bins (no cut-points chosen), each with mean |offset| in px and the
    same in metres, |offset| z / f. Slopes need no floor and no log: y on z by least squares, for y = |offset| (px) and
    y = |offset| z / f (m). Constant in pixels <=> the px slope is 0; constant in metres <=> the m slope is 0. Intervals:
    moving-block bootstrap over the frames in time order, block length from the autocorrelation of the |offset| series
    (block_length). Frames within a block stay together, so the interval is not tightened by the strong correlation of
    consecutive frames."""
    out = {}
    for c in sorted({r['cls'] for r in rows}):
        by = {}
        for r in rows:
            if r['cls'] == c: by.setdefault(r['stamp_ms'], []).append(r)
        stamps = sorted(by)
        z = np.array([np.mean([r['range_m'] for r in by[t]]) for t in stamps])
        o = np.array([abs(np.mean([r['mean_offset_px'] for r in by[t]])) for t in stamps])
        if len(z) < 2 * n_bins: out[c] = None; continue
        m = o * z / f
        order = np.argsort(z); bins = []
        for q in np.array_split(order, n_bins):
            bins.append((z[q].min(), z[q].max(), len(q), o[q].mean(), m[q].mean()))
        L, r1 = block_length(o)
        out[c] = dict(bins=bins, block=L, acf1=r1, n=len(z), px=block_bootstrap_slope(z, o, L, n_boot, seed),
                      m=block_bootstrap_slope(z, m, L, n_boot, seed))
    return out


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dump_dir'); ap.add_argument('wbt'); ap.add_argument('slam_shadow', nargs='?')
    ap.add_argument('--classes', default='nightstand,bed,toilet')
    ap.add_argument('--extents', nargs='*', default=[], help='class=lx,ly,z0,z1[,ox,oy] (m, object frame; box centre at ox,oy, default 0,0)')
    ap.add_argument('--hfov', type=float, default=1.9198)
    ap.add_argument('--out')
    a = ap.parse_args(argv)
    classes = a.classes.split(',')
    extents = dict(KNOWN_EXTENTS)
    extents.update({k: tuple(map(float, v.split(','))) for k, v in (e.split('=') for e in a.extents)})
    rows = measure(a.dump_dir, a.wbt, classes, extents, a.hfov)
    if a.out:
        with open(a.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['stamp_ms']); w.writeheader(); w.writerows(rows)
    print('class | n frames | n masks | sigma_px_boundary | mean of frame means | eps_px | median residual std')
    for c, s in summarise(rows).items():
        print(f"{c} | {s['n_frames']} | {s['n_masks']} | {s['sigma_px_boundary']:.2f} | {s['mean_of_frame_means_px']:.2f} | "
              f"{s['eps_px']:.3f} | {s['median_resid_std_px']:.2f}")
    for c, rb in range_bins(rows, intrinsics(1280, 720, a.hfov)[0]).items():
        print(f'range bins, {c}: z range (m) | n frames | mean |offset| px | mean |offset|*z/f m')
        if rb is None: print('  too few frames'); continue
        for lo, hi, n, px, m in rb['bins']: print(f'  {lo:.2f}-{hi:.2f} | {n} | {px:.2f} | {m:.4f}')
        print(f"  n {rb['n']}, block length {rb['block']} frames (acf at lag 1 = {rb['acf1']:.2f}); OLS slope of y on z, 95% block bootstrap:")
        for k, unit in (('px', 'px per m'), ('m', 'm per m')):
            b0, lo, hi = rb[k]
            print(f"    y = |offset| {'(px)' if k == 'px' else 'z/f (m)'}: {b0:+.4f} {unit} [{lo:+.4f}, {hi:+.4f}]  contains 0: {lo <= 0 <= hi}")
    if not rows: print('no mask matched a graded box (no frame with gt_valid=1, or no overlap)')


if __name__ == '__main__':
    main(sys.argv[1:])
