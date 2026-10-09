"""Planar rigid motions. A pose is a length-3 array (x, y, theta)."""
import numpy as np


def wrap(a):
    """Wrap an angle (or array of angles) to (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def compose(a, b):
    """a ∘ b: express pose b (given in frame a) in a's parent frame."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    xy = a[:2] + rot(a[2]) @ b[:2]
    return np.array([xy[0], xy[1], wrap(a[2] + b[2])])


def inverse(a):
    a = np.asarray(a, float)
    xy = -rot(a[2]).T @ a[:2]
    return np.array([xy[0], xy[1], wrap(-a[2])])


def ominus(a, b):
    """a ⊖ b: pose a expressed in frame b."""
    return compose(inverse(b), a)


def transform_points(pose, pts):
    """Map points (N, 2) from the frame `pose` into its parent frame."""
    pts = np.atleast_2d(pts)
    return pts @ rot(pose[2]).T + np.asarray(pose[:2])


def to_frame(pose, pts):
    """Map points (N, 2) from the parent frame into the frame `pose`."""
    pts = np.atleast_2d(pts)
    return (pts - np.asarray(pose[:2])) @ rot(pose[2])
