import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toyworld import Gaussian, Graph, Lidar, Robot, info_gain, laplace, se2, two_room_apartment  # noqa: E402
from toyworld.beliefs import entropy, kl, linear_gaussian_update  # noqa: E402
from toyworld.geometry import distance_to_polygon, polygon_iou  # noqa: E402
from toyworld.robot import Body, integrate  # noqa: E402


def test_se2_roundtrip():
    a, b = np.array([1.0, -2.0, 0.7]), np.array([0.3, 0.4, -2.9])
    assert np.allclose(se2.ominus(se2.compose(a, b), a), b)
    pts = np.random.default_rng(0).normal(size=(5, 2))
    assert np.allclose(se2.to_frame(a, se2.transform_points(a, pts)), pts)


def test_lidar_hits_walls_where_expected():
    world = two_room_apartment()
    lidar = Lidar(n_beams=4, sigma=0.0)
    scan = lidar.scan(world, [3.0, 1.0, 0.0])   # beams at -pi, -pi/2, 0, pi/2
    # -pi: west wall at x=0 (3 m); -pi/2: south wall (1 m); 0: east wall x=6 is solid at y=1 (3 m);
    # +pi/2: table spans y∈[2.2, 3.0] only for x∈[1.6, 2.8], so x=3 reaches the north wall (3 m).
    assert np.allclose(scan.ranges, [3.0, 1.0, 3.0, 3.0])
    assert list(scan.labels) == ["wall"] * 4


def test_lidar_sees_through_the_door_and_hits_objects():
    world = two_room_apartment()
    lidar = Lidar(n_beams=4, sigma=0.0)
    through = lidar.scan(world, [5.0, 2.0, 0.0])       # beam 0 goes through the door gap at y=2
    assert np.isclose(through.ranges[2], 5.0)          # to the bedroom's east wall at x=10
    at_table = lidar.scan(world, [2.2, 1.0, 0.0])       # beam +pi/2 hits the table's south face
    assert np.isclose(at_table.ranges[3], 2.2 - 1.0) and at_table.labels[3] == "table"


def test_lidar_noise_is_independent_unless_asked():
    world, lidar = two_room_apartment(), Lidar(sigma=0.02)
    rng = np.random.default_rng(1)
    s = lidar.scan(world, [3, 2, 0], rng)
    err = (s.ranges - s.true_ranges)[s.hit]
    assert abs(err.std() - 0.02) < 0.004 and abs(err.mean()) < 0.004
    s2 = lidar.scan(world, [3, 2, 0], rng, common_offset=0.1)
    assert abs(np.nanmean(s2.ranges - s2.true_ranges) - 0.1) < 0.004


def test_odometry_reports_scaled_speed_and_biased_gyro():
    body = Body(wheel_scale=1.04, gyro_bias=0.01, slip_v=0, slip_w=0, enc_sigma=0, gyro_sigma=0)
    robot = Robot([0, 0, 0], body)
    odo = robot.step(0.5, 0.0, dt=1.0)
    assert np.isclose(odo.v, 0.52) and np.isclose(odo.w, 0.01)
    assert np.allclose(robot.pose, [0.5, 0, 0])


def test_unicycle_closes_a_circle():
    pose = np.zeros(3)
    for _ in range(100):
        pose = integrate(pose, 0.2 * np.pi, 0.2 * np.pi, 0.1)   # one full turn in 10 s
    assert np.allclose(pose, 0, atol=1e-9)


def _wall_points(rng, n=40, sigma=0.02):
    x = rng.uniform(1, 5, n)
    return np.stack([x, 4.0 + sigma * rng.standard_normal(n)], axis=1)


def _wall_residual(pts):
    return lambda p: np.cos(p[0]) * pts[:, 0] + np.sin(p[0]) * pts[:, 1] - p[1]


def test_laplace_covariance_matches_monte_carlo():
    """The Laplace Σ = H⁻¹ of a wall fit must match the spread of the fit across noise draws."""
    rng = np.random.default_rng(2)
    fits, sigmas = [], []
    for _ in range(1500):
        res = laplace(_wall_residual(_wall_points(rng)), [np.pi / 2 + 0.05, 3.9], 0.02)
        fits.append(res.belief.mu)
        sigmas.append(res.belief.Sigma)
    empirical, predicted = np.cov(np.array(fits).T), np.mean(sigmas, axis=0)
    assert np.allclose(np.sqrt(np.diag(empirical)), np.sqrt(np.diag(predicted)), rtol=0.08)


def test_laplace_free_energy_prefers_the_true_model():
    """Model comparison by Laplace F: a line beats a constant for sloped data."""
    rng = np.random.default_rng(3)
    x = np.linspace(0, 1, 50)
    y = 0.5 * x + 0.02 * rng.standard_normal(50)
    flat = laplace(lambda p: y - p[0], [0.0], 0.02, prior=Gaussian([0], [[1]]))
    line = laplace(lambda p: y - p[0] - p[1] * x, [0.0, 0.0], 0.02, prior=Gaussian([0, 0], np.eye(2)))
    assert line.free_energy < flat.free_energy - 10


def test_shared_offset_sets_a_ceiling():
    """N readings sharing a common-mode error c: var(mean) = c² + σ²/N, never → 0."""
    sigma, c = 0.02, 0.05
    for n in (10, 100, 1000):
        R = sigma ** 2 * np.eye(n) + c ** 2 * np.ones((n, n))
        post = linear_gaussian_update(Gaussian([0], [[1e6]]), np.ones((n, 1)), R, np.zeros(n))
        assert np.isclose(post.Sigma[0, 0], c ** 2 + sigma ** 2 / n, rtol=1e-4)


def test_recovery_identity():
    """Regressing the posterior mean on an injected truth gives slope 1 − σ_post²/σ_prior²."""
    rng = np.random.default_rng(4)
    sp, sn = 0.03, 0.02
    truth = sp * rng.standard_normal(20000)
    y = truth + sn * rng.standard_normal(20000)
    prior = Gaussian([0], [[sp ** 2]])
    post_var = linear_gaussian_update(prior, [[1]], [[sn ** 2]], [0]).Sigma[0, 0]
    est = post_var / sn ** 2 * y
    slope = np.polyfit(truth, est, 1)[0]
    assert np.isclose(slope, 1 - post_var / sp ** 2, atol=0.01)


def test_info_gain_is_entropy_drop():
    prior = Gaussian([0, 0], np.diag([0.04, 0.09]))
    J, R = np.array([[1.0, 0.5]]), np.array([[0.01]])
    post = linear_gaussian_update(prior, J, R, [0.0])
    assert np.isclose(info_gain(prior.precision, J, R), entropy(prior.Sigma) - entropy(post.Sigma))
    assert np.isclose(kl(prior, prior), 0.0)


def test_geometry():
    square = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], float)
    assert np.allclose(distance_to_polygon([[1, 1], [3, 1]], square), [1, 1])
    assert np.isclose(polygon_iou(square, square + [1, 0]), 1 / 3, atol=0.02)


def test_graph_transforms_and_removal():
    g = Graph()
    room = g.add_node("living", "room", parent="world", pose=[1, 0, np.pi / 2])
    robot = g.add_node("robot", "robot", parent=room, pose=[2, 0, 0])
    g.add_node("table", "table", parent=room, pose=[0, 3, 0])
    assert np.allclose(g.world_pose(robot), [1, 2, np.pi / 2])
    assert np.allclose(g.transform(robot, "table"), [-2, 3, 0])
    g.add_edge("robot", "table", "sees")
    g.remove("living")
    assert [n.name for n in g.nodes.values()] == ["world"] and not g.edges


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
