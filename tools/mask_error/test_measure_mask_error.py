#!/usr/bin/env python3
"""Tests for measure_mask_error.py.  Run: python3 tools/mask_error/test_measure_mask_error.py

The main test builds a synthetic mask edge: a rectangle silhouette whose edge is moved OUTWARD by a known 3 px, with
independent Gaussian noise of known std (1.5 px) on every contour sample, and checks that the tool recovers 3 px as the
per-frame mean offset and 1.5 px as the residual std.

HOW THE TOLERANCES ARE DERIVED (nothing is fitted to the run):
  n independent contour samples, each offset = shift + e, e ~ N(0, s^2).
  * the sample mean has standard error s / sqrt(n);
  * the sample std has standard error about s / sqrt(2 (n - 1)) (large-n result for a normal sample).
  Both tolerances are 4 standard errors, i.e. a false failure probability near 6e-5 per assertion. With the fixed seed
  below the draw is deterministic, so this is the bound a different seed would also meet 99.99% of the time.
  * For the raster path (a mask rebuilt from decimated pixels with stride 2) the edge position is quantised to the
    block size, so each edge is off by at most stride/2 = 1 px; the mean offset is then within stride/2 of the shift,
    and every offset lies within stride/2 of its mean edge, so the residual std is at most stride/2.
"""
import math, os, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure_mask_error as M


def rect_hull(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32).reshape(-1, 1, 2)


def edge_samples(x0, y0, x1, y1, per_edge, rng):
    """Points on the interior of each edge (corners excluded), with the outward unit normal of that edge."""
    pts, nrm = [], []
    for (ax, ay, bx, by, nx, ny) in ((x0, y0, x1, y0, 0, -1), (x1, y0, x1, y1, 1, 0),
                                     (x1, y1, x0, y1, 0, 1), (x0, y1, x0, y0, -1, 0)):
        t = rng.uniform(0.05, 0.95, per_edge)
        pts += list(zip(ax + t * (bx - ax), ay + t * (by - ay))); nrm += [(nx, ny)] * per_edge
    return np.array(pts), np.array(nrm, float)


def test_recovers_shift_and_noise():
    rng = np.random.default_rng(20261124)
    shift, s, per_edge = 3.0, 1.5, 150
    n = 4 * per_edge
    hull = rect_hull(100, 200, 400, 320)
    p, nrm = edge_samples(100, 200, 400, 320, per_edge, rng)
    samples = p + nrm * (shift + rng.normal(0.0, s, n))[:, None]
    off = M.signed_offsets(samples, hull)
    se_mean, se_std = s / math.sqrt(n), s / math.sqrt(2 * (n - 1))
    assert abs(off.mean() - shift) <= 4 * se_mean, (off.mean(), shift, 4 * se_mean)
    assert abs(off.std(ddof=1) - s) <= 4 * se_std, (off.std(ddof=1), s, 4 * se_std)
    # frame_stats must report the same numbers plus the half-pixel pixel-centre correction, nothing else
    mean, std = M.frame_stats(off)
    assert abs(mean - (off.mean() + M.HALF_PX)) < 1e-9 and abs(std - off.std(ddof=1)) < 1e-9


def test_sign_inside_is_negative():
    hull = rect_hull(100, 200, 400, 320)
    o = M.signed_offsets(np.array([[250.0, 210.0], [250.0, 190.0]]), hull)
    assert abs(o[0] + 10) < 1e-9 and abs(o[1] - 10) < 1e-9, o


def test_raster_path_grown_mask():
    stride, shift = 2, 3
    x0, y0, x1, y1 = 100, 200, 400, 320
    hull = rect_hull(x0, y0, x1, y1)
    pix = [(c, r) for c in range(0, 640, stride) for r in range(0, 480, stride)
           if x0 - shift <= c + stride / 2 - 0.5 <= x1 + shift and y0 - shift <= r + stride / 2 - 0.5 <= y1 + shift]
    mk = M.mask_raster(pix, stride, 480, 640)
    cs = M.contour_samples(mk)
    mean, std = M.frame_stats(M.signed_offsets(cs, hull))
    assert abs(mean - shift) <= stride / 2, mean
    assert std <= stride / 2, std
    # a mask that exactly fills the silhouette has no pixel outside it (the .5 px band at the edge counts as inside)
    pix0 = [(c, r) for c in range(x0, x1, stride) for r in range(y0, y1, stride)]
    assert M.fraction_outside(M.mask_raster(pix0, stride, 480, 640), hull) == 0.0


def test_projection_symmetric_ahead():
    K = M.intrinsics(1280, 720, 1.9198)
    assert abs(K[0] - 448.1) < 0.2 and K[2] == 640 and K[3] == 360
    # robot at the origin facing +y: gt_angle = -(yaw + pi/2) = -pi/2 for yaw 0; camera 0.075 m behind it
    cam = M.camera_pose(0.0, 0.0, -math.pi / 2, 0.9775)
    assert abs(cam[0]) < 1e-9 and abs(cam[1] + 0.075) < 1e-9 and abs(cam[3] - math.pi / 2) < 1e-9
    px = M.project(M.box_corners(0.0, 3.0, 0.0, 0.5, 0.5, 0.0, 0.6), cam, M.intrinsics(1280, 720, 1.9198))
    assert abs(px[:, 0].min() + px[:, 0].max() - 1280) < 1e-6            # centred on the principal column
    assert M.project(M.box_corners(0.0, -1.0, 0.0, 0.5, 0.5, 0.0, 0.6), cam, K) is None   # behind the camera


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn(); print('ok', name)
