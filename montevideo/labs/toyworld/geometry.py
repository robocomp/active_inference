"""Polygons and segments: distances, containment, area and IoU."""
import numpy as np


def segments_of(polygon):
    """Closed polygon (N, 2) -> array of segments (N, 2, 2)."""
    p = np.asarray(polygon, float)
    return np.stack([p, np.roll(p, -1, axis=0)], axis=1)


def point_segment_distance(pts, segs):
    """Distance from each point (N, 2) to each segment (M, 2, 2) -> (N, M)."""
    pts = np.atleast_2d(pts)[:, None, :]
    a, b = segs[None, :, 0, :], segs[None, :, 1, :]
    ab = b - a
    t = np.sum((pts - a) * ab, axis=-1) / np.maximum(np.sum(ab * ab, axis=-1), 1e-12)
    t = np.clip(t, 0.0, 1.0)
    closest = a + t[..., None] * ab
    return np.linalg.norm(pts[..., :] - closest, axis=-1)


def distance_to_polygon(pts, polygon):
    """Unsigned distance from each point to the nearest polygon edge: the SDF magnitude."""
    return point_segment_distance(pts, segments_of(polygon)).min(axis=1)


def point_in_polygon(pts, polygon):
    """Even-odd rule, vectorised over points."""
    pts = np.atleast_2d(pts)
    poly = np.asarray(polygon, float)
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    x0, y0 = poly[:, 0][None], poly[:, 1][None]
    x1, y1 = np.roll(poly[:, 0], -1)[None], np.roll(poly[:, 1], -1)[None]
    crosses = ((y0 > y) != (y1 > y)) & (x < (x1 - x0) * (y - y0) / np.where(y1 == y0, 1e-12, y1 - y0) + x0)
    return np.count_nonzero(crosses, axis=1) % 2 == 1


def polygon_area(polygon):
    p = np.asarray(polygon, float)
    return 0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1)))


def polygon_iou(a, b, resolution=0.02):
    """IoU of two polygons by rasterisation on a common grid."""
    both = np.vstack([a, b])
    lo, hi = both.min(axis=0), both.max(axis=0)
    xs = np.arange(lo[0], hi[0] + resolution, resolution)
    ys = np.arange(lo[1], hi[1] + resolution, resolution)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    ia, ib = point_in_polygon(grid, a), point_in_polygon(grid, b)
    union = np.count_nonzero(ia | ib)
    return np.count_nonzero(ia & ib) / union if union else 0.0
