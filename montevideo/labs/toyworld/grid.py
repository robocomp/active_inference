"""A log-odds occupancy grid, and the information a look from a viewpoint is expected to buy.

Day 3 builds the explorer on it; Day 5 reuses it to finish room 1 and map room 2.
"""
import numpy as np

from .beliefs import binary_entropy, sigmoid
from .geometry import point_in_polygon
from .lidar import raycast


def cell_gain(l, l_hit, l_miss):
    """Expected entropy drop of a cell with log-odds l, averaged over the two outcomes of a look:
    ΔH = H(p) − [p H(σ(l + l_hit)) + (1 − p) H(σ(l − l_miss))]."""
    p = sigmoid(l)
    return binary_entropy(p) - (p * binary_entropy(sigmoid(l + l_hit))
                                + (1 - p) * binary_entropy(sigmoid(l - l_miss)))


class Grid:
    def __init__(self, lo=(0.0, 0.0), hi=(6.0, 4.0), res=0.05, l_hit=2.2, l_miss=2.2, band=0.05):
        self.lo, self.res = np.asarray(lo, float), res
        self.shape = tuple(np.ceil((np.asarray(hi) - self.lo) / res).astype(int))
        self.l = np.zeros(self.shape)              # log-odds 0: p = 0.5, nothing known
        self.l_hit, self.l_miss = l_hit, l_miss
        self.band = band                           # cells this close to a return are not "free"
        ix, iy = np.meshgrid(np.arange(self.shape[0]), np.arange(self.shape[1]), indexing="ij")
        self.centres = self.lo + (np.stack([ix, iy], -1) + 0.5) * res

    @property
    def p(self):
        return sigmoid(self.l)

    def mask(self, polygon):
        """Cells whose centre lies inside `polygon`."""
        return point_in_polygon(self.centres.reshape(-1, 2), polygon).reshape(self.shape)

    def entropy(self, mask=None):
        h = binary_entropy(self.p)
        return float(h.sum() if mask is None else h[mask].sum())

    def march(self, origin, angles, ranges):
        """Cells crossed by each beam, sampled every half cell: (beam, i, j, t), t = distance."""
        R = np.nan_to_num(np.asarray(ranges, float), nan=0.0, posinf=0.0)
        step = self.res / 2
        n = np.ceil(R / step).astype(int)
        B = np.repeat(np.arange(len(angles)), n + 1)
        k = np.concatenate([np.arange(m + 1) for m in n]) if len(n) else np.zeros(0, int)
        t = np.minimum(k * step, R[B])
        pts = np.asarray(origin, float)[None] + t[:, None] * np.stack([np.cos(angles[B]), np.sin(angles[B])], 1)
        ij = np.floor((pts - self.lo) / self.res).astype(int)
        ok = (ij >= 0).all(1) & (ij[:, 0] < self.shape[0]) & (ij[:, 1] < self.shape[1])
        return B[ok], ij[ok, 0], ij[ok, 1], t[ok]

    def touched(self, origin, angles, ranges):
        """What one look does to the grid: cells passed through (free) and cells at a return (hit).

        Cells within `band` of the return are neither: range noise could put the surface in them.
        """
        B, i, j, t = self.march(origin, angles, ranges)
        R = np.asarray(ranges, float)[B]
        free, hit = np.zeros(self.shape, bool), np.zeros(self.shape, bool)
        f = t < R - self.band
        free[i[f], j[f]] = True
        e = t >= R - 1e-9
        hit[i[e], j[e]] = True
        return free & ~hit, hit

    def update(self, scan):
        """Fold one scan into the grid (the scan's pose is the sensor pose, world frame)."""
        fin = np.isfinite(scan.ranges)
        free, hit = self.touched(scan.pose[:2], scan.pose[2] + scan.angles[fin], scan.ranges[fin])
        self.l[free] -= self.l_miss
        self.l[hit] += self.l_hit

    def matter(self):
        """Edges of the cells the grid believes occupied (p > ½), as segments for ray casting."""
        occ = np.argwhere(self.l > 0)
        if not len(occ):
            return np.zeros((0, 2, 2))
        c, h = self.lo + (occ + 0.5) * self.res, self.res / 2
        sq = np.stack([c + [-h, -h], c + [h, -h], c + [h, h], c + [-h, h]], 1)
        return np.concatenate([np.stack([sq[:, k], sq[:, (k + 1) % 4]], 1) for k in range(4)])


def expected_gain(grid, viewpoint, walls, mask, through_matter=False, n_beams=180):
    """Nats a 360° look from `viewpoint` is expected to remove from the cells in `mask`.

    Beams are predicted through the model: the room's walls and, unless `through_matter`, every
    cell the grid already believes occupied. Each cell the predicted look would touch contributes
    its `cell_gain`. Exactly the cells `Grid.touched` would update are counted.
    """
    ang = np.linspace(-np.pi, np.pi, n_beams, endpoint=False)
    dirs = np.stack([np.cos(ang), np.sin(ang)], 1)
    segs = walls if through_matter else np.concatenate([walls, grid.matter()])
    r, _ = raycast(viewpoint, dirs, segs, np.zeros((0, 2)), np.zeros(0))
    free, hit = grid.touched(viewpoint, ang, r)
    return float(cell_gain(grid.l, grid.l_hit, grid.l_miss)[(free | hit) & mask].sum())
