"""A differential-drive robot with a body that is not quite what its designers wrote down.

The robot executes velocity commands. It reports what its encoders and gyro measured.
Those reports carry systematic errors the robot does not know about: a wheel-scale factor, a
gyro scale factor and a gyro bias. Day 4 is about inferring them.
"""
from dataclasses import dataclass

import numpy as np

from . import se2


def integrate(pose, v, w, dt):
    """Exact unicycle motion over dt at constant (v, w)."""
    x, y, th = pose
    if abs(w) < 1e-9:
        return np.array([x + v * dt * np.cos(th), y + v * dt * np.sin(th), th])
    th1 = th + w * dt
    return np.array([x + v / w * (np.sin(th1) - np.sin(th)),
                     y - v / w * (np.cos(th1) - np.cos(th)),
                     se2.wrap(th1)])


@dataclass
class Body:
    wheel_scale: float = 1.0   # encoders report wheel_scale × the true forward speed
    gyro_scale: float = 1.0    # the gyro reports gyro_scale × the true turn rate
    gyro_bias: float = 0.0     # rad/s added to every gyro reading
    slip_v: float = 0.02       # true speed deviates from the command, sd per (m/s)
    slip_w: float = 0.02       # true turn rate deviates from the command, sd per (rad/s)
    enc_sigma: float = 0.005   # encoder noise on v (m/s)
    gyro_sigma: float = 0.003  # gyro noise on w (rad/s)


@dataclass
class Odometry:
    v: float   # measured forward speed
    w: float   # measured turn rate
    dt: float


class Robot:
    def __init__(self, pose, body=None, rng=None):
        self.pose = np.asarray(pose, float)   # TRUE pose; the robot itself never reads this
        self.body = Body() if body is None else body
        self.rng = np.random.default_rng() if rng is None else rng

    def step(self, v_cmd, w_cmd, dt=0.1):
        """Move under a command; return what the robot's own sensors measured."""
        b, n = self.body, self.rng.standard_normal(4)
        v = v_cmd * (1 + b.slip_v * n[0])
        w = w_cmd + b.slip_w * abs(w_cmd) * n[1]
        self.pose = integrate(self.pose, v, w, dt)
        return Odometry(v * b.wheel_scale + b.enc_sigma * n[2],
                        w * b.gyro_scale + b.gyro_bias + b.gyro_sigma * n[3], dt)


def motion_jacobians(pose, v, w, dt, eps=1e-6):
    """∂integrate/∂pose (3x3) and ∂integrate/∂(v, w) (3x2), by central differences."""
    pose = np.asarray(pose, float)
    F, G = np.empty((3, 3)), np.empty((3, 2))
    for i in range(3):
        d = np.zeros(3)
        d[i] = eps
        diff = integrate(pose + d, v, w, dt) - integrate(pose - d, v, w, dt)
        diff[2] = se2.wrap(diff[2])
        F[:, i] = diff / (2 * eps)
    for i, du in enumerate(([eps, 0], [0, eps])):
        diff = integrate(pose, v + du[0], w + du[1], dt) - integrate(pose, v - du[0], w - du[1], dt)
        diff[2] = se2.wrap(diff[2])
        G[:, i] = diff / (2 * eps)
    return F, G


def predict(belief, odo, sigma_v=0.005, sigma_w=0.003):
    """Propagate a pose belief through one odometry reading.

    The reading's own noise (encoder sd sigma_v in m/s, gyro sd sigma_w in rad/s) is pushed through
    the motion's Jacobian: Σ' = F Σ Fᵀ + G diag(σ_v², σ_w²) Gᵀ. No constant process noise is added:
    the uncertainty grows because of what the sensors cannot tell, not by decree.
    """
    from .beliefs import Gaussian
    F, G = motion_jacobians(belief.mu, odo.v, odo.w, odo.dt)
    mu = integrate(belief.mu, odo.v, odo.w, odo.dt)
    return Gaussian(mu, F @ belief.Sigma @ F.T + G @ np.diag([sigma_v ** 2, sigma_w ** 2]) @ G.T)


def hidden_body():
    """The body Day 4 has to discover. Do not read this cell before the lab!"""
    return Body(wheel_scale=1.04, gyro_scale=1.03, gyro_bias=0.012)
