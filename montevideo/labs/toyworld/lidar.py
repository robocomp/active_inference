"""A planar LiDAR: exact ray casting against the world's segments and discs, plus range noise."""
from dataclasses import dataclass

import numpy as np

from . import se2


@dataclass
class Scan:
    pose: np.ndarray          # the TRUE sensor pose (world frame) — for grading only
    angles: np.ndarray        # beam bearings in the sensor frame
    ranges: np.ndarray        # measured ranges; nan where nothing was hit within range
    labels: np.ndarray        # what each beam hit ('wall', object name, or '' for no hit)
    true_ranges: np.ndarray   # noiseless ranges — for grading only

    @property
    def hit(self):
        return np.isfinite(self.ranges)

    def points(self):
        """Measured hit points in the SENSOR frame (K, 2). This is what the robot gets."""
        h = self.hit
        r, a = self.ranges[h], self.angles[h]
        return np.stack([r * np.cos(a), r * np.sin(a)], axis=1)


def _cross(u, v):
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def raycast(origin, directions, segs, centres, radii):
    """Distance along each unit direction (B, 2) to the first segment or disc; inf if none.

    Returns (distance (B,), index (B,)). Indices < len(segs) are segments; the rest are discs.
    """
    o = np.asarray(origin, float)
    d = directions[:, None, :]                       # (B, 1, 2)
    best = np.full(len(directions), np.inf)
    idx = np.full(len(directions), -1)
    if len(segs):
        p, e = segs[None, :, 0, :], (segs[:, 1, :] - segs[:, 0, :])[None]
        denom = _cross(d, e)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = _cross(p - o, e) / denom
            u = _cross(p - o, d) / denom
        t = np.where((np.abs(denom) > 1e-12) & (t > 1e-9) & (u >= 0) & (u <= 1), t, np.inf)
        best, idx = t.min(axis=1), t.argmin(axis=1)
    if len(centres):
        oc = centres[None] - o                       # (1, K, 2)
        b = np.sum(d * oc, axis=-1)                  # (B, K)
        disc = b * b - (np.sum(oc * oc, axis=-1) - radii[None] ** 2)
        with np.errstate(invalid="ignore"):
            t = b - np.sqrt(disc)
        t = np.where((disc >= 0) & (t > 1e-9), t, np.inf)
        tc, ic = t.min(axis=1), t.argmin(axis=1)
        closer = tc < best
        best = np.where(closer, tc, best)
        idx = np.where(closer, len(segs) + ic, idx)
    return best, idx


@dataclass
class Lidar:
    n_beams: int = 360
    fov: float = 2 * np.pi
    max_range: float = 12.0
    sigma: float = 0.02          # per-beam range noise (m), independent across beams
    mount: tuple = (0.0, 0.0, 0.0)  # sensor pose in the robot frame

    def angles(self):
        if np.isclose(self.fov, 2 * np.pi):
            return np.linspace(-np.pi, np.pi, self.n_beams, endpoint=False)
        return np.linspace(-self.fov / 2, self.fov / 2, self.n_beams)

    def scan(self, world, robot_pose, rng=None, common_offset=0.0):
        """Scan from robot_pose (world frame).

        `common_offset` adds the SAME range error to every beam — a shared, not independent,
        error. Leave it at 0 unless an exercise asks for it.
        """
        rng = np.random.default_rng() if rng is None else rng
        pose = se2.compose(robot_pose, self.mount)
        angles = self.angles()
        world_angles = pose[2] + angles
        dirs = np.stack([np.cos(world_angles), np.sin(world_angles)], axis=1)
        segs, seg_labels = world.segments()
        centres, radii, disc_labels = world.discs()
        true_r, idx = raycast(pose[:2], dirs, segs, centres, radii)
        all_labels = np.concatenate([seg_labels, disc_labels]).astype(str)
        hit = np.isfinite(true_r) & (true_r <= self.max_range)
        labels = np.where(hit, all_labels[np.clip(idx, 0, None)] if len(all_labels) else "", "")
        noisy = true_r + common_offset + self.sigma * rng.standard_normal(len(true_r))
        ranges = np.where(hit, noisy, np.nan)
        return Scan(pose, angles, ranges, labels, np.where(hit, true_r, np.nan))
