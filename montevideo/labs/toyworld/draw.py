"""Matplotlib helpers shared by every lab."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, Polygon

from . import se2


def world(w, ax=None, things=True):
    ax = ax or plt.gca()
    for room in w.rooms:
        ax.add_patch(Polygon(room.polygon, closed=True, fc="0.96", ec="none"))
        ax.text(*room.polygon.mean(axis=0), room.name, ha="center", va="center", color="0.6", clip_on=True)
    for room in w.rooms:
        for a, b in w.room_walls(room.name):
            ax.plot([a[0], b[0]], [a[1], b[1]], color="0.15", lw=2.5, solid_capstyle="butt")
    if things:
        for t in w.things:
            if t.kind == "box":
                ax.add_patch(Polygon(t.outline(), fc="tan", ec="saddlebrown"))
            else:
                ax.add_patch(Circle(t.pose[:2], t.size[0], fc="lightsteelblue", ec="steelblue"))
    ax.set_aspect("equal")
    ax.autoscale_view()
    return ax


def robot(pose, ax=None, color="crimson", size=0.25, **kw):
    ax = ax or plt.gca()
    tri = np.array([[size, 0], [-size * 0.6, size * 0.5], [-size * 0.6, -size * 0.5]])
    ax.add_patch(Polygon(se2.transform_points(pose, tri), fc=color, ec="none", **kw))
    return ax


def scan(s, pose=None, ax=None, rays=False, **kw):
    """Plot a scan's points in the world frame, placed at `pose` (defaults to the true pose)."""
    ax = ax or plt.gca()
    pose = s.pose if pose is None else pose
    pts = se2.transform_points(pose, s.points())
    if rays:
        for p in pts:
            ax.plot([pose[0], p[0]], [pose[1], p[1]], color="crimson", lw=0.3, alpha=0.3)
    ax.plot(pts[:, 0], pts[:, 1], ".", ms=3, color=kw.pop("color", "crimson"), **kw)
    return ax


def covariance(mu, Sigma, ax=None, n_sigma=2.0, **kw):
    """The n-sigma ellipse of a 2-D Gaussian."""
    ax = ax or plt.gca()
    vals, vecs = np.linalg.eigh(np.asarray(Sigma)[:2, :2])
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    w, h = 2 * n_sigma * np.sqrt(np.maximum(vals[::-1], 0))
    kw.setdefault("fc", "none")
    kw.setdefault("ec", "royalblue")
    ax.add_patch(Ellipse(mu[:2], w, h, angle=angle, **kw))
    return ax
