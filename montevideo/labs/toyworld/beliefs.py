"""Gaussian beliefs, the Laplace approximation, and the information arithmetic of the course."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Gaussian:
    mu: np.ndarray
    Sigma: np.ndarray

    def __post_init__(self):
        self.mu = np.atleast_1d(np.asarray(self.mu, float))
        self.Sigma = np.atleast_2d(np.asarray(self.Sigma, float))

    @property
    def precision(self):
        return np.linalg.inv(self.Sigma)

    @property
    def sd(self):
        return np.sqrt(np.diag(self.Sigma))

    def entropy(self):
        return entropy(self.Sigma)

    def sample(self, n, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        return rng.multivariate_normal(self.mu, self.Sigma, size=n)


def entropy(Sigma):
    """Differential entropy of a Gaussian, in nats: ½ ln |2πe Σ|."""
    Sigma = np.atleast_2d(Sigma)
    return 0.5 * np.linalg.slogdet(2 * np.pi * np.e * Sigma)[1]


def kl(q, p):
    """KL[q || p] between two Gaussians, in nats."""
    k = len(q.mu)
    Pinv = np.linalg.inv(p.Sigma)
    dm = p.mu - q.mu
    return 0.5 * (np.trace(Pinv @ q.Sigma) + dm @ Pinv @ dm - k
                  + np.linalg.slogdet(p.Sigma)[1] - np.linalg.slogdet(q.Sigma)[1])


def info_gain(Lambda, J, R):
    """Expected entropy reduction (nats) from a linear-Gaussian measurement.

    Prior precision Lambda, measurement Jacobian J, measurement noise covariance R:
    ΔH = ½ ln |Λ + JᵀR⁻¹J| − ½ ln |Λ|. This is also the mutual information between the
    state and the measurement, and the D-optimality criterion of experiment design.
    """
    J, R = np.atleast_2d(J), np.atleast_2d(R)
    post = Lambda + J.T @ np.linalg.solve(R, J)
    return 0.5 * (np.linalg.slogdet(post)[1] - np.linalg.slogdet(Lambda)[1])


def numeric_jacobian(f, x, eps=1e-6):
    x = np.asarray(x, float)
    f0 = np.atleast_1d(f(x))
    J = np.empty((len(f0), len(x)))
    for i in range(len(x)):
        dx = np.zeros_like(x)
        dx[i] = eps
        J[:, i] = (np.atleast_1d(f(x + dx)) - np.atleast_1d(f(x - dx))) / (2 * eps)
    return J


@dataclass
class LaplaceResult:
    belief: Gaussian
    free_energy: float   # Laplace approximation to −ln p(o): the model's surprise
    cost: float          # −ln p(o, μ) at the optimum (accuracy + complexity, unnormalised by H)
    iterations: int


def laplace(residual, x0, sigma, prior=None, jacobian=None, iters=50, tol=1e-10):
    """Gauss–Newton MAP fit with a Laplace posterior.

    residual(x) -> r (N,), whitened by `sigma` (scalar or (N,) array): the likelihood is
    N(r; 0, diag(sigma²)). `prior` is an optional Gaussian on x. Returns the belief
    N(μ, H⁻¹) and the Laplace free energy
        F ≈ −ln p(o, μ) + ½ ln |H| − (n/2) ln 2π,
    which is what model comparison (birth, shape selection) needs.
    """
    jac = jacobian or (lambda x: numeric_jacobian(residual, x))
    x = np.asarray(x0, float).copy()
    n = len(x)

    def terms(x):
        r = np.atleast_1d(residual(x))
        s = np.broadcast_to(np.asarray(sigma, float), r.shape)
        c = 0.5 * np.sum((r / s) ** 2) + np.sum(np.log(s)) + 0.5 * len(r) * np.log(2 * np.pi)
        if prior is not None:
            d = x - prior.mu
            c += 0.5 * d @ prior.precision @ d + 0.5 * np.linalg.slogdet(2 * np.pi * prior.Sigma)[1]
        return r, s, c

    it = 0
    for it in range(1, iters + 1):
        r, s, c = terms(x)
        J = np.atleast_2d(jac(x)) / s[:, None]
        H = J.T @ J
        g = J.T @ (r / s)
        if prior is not None:
            H = H + prior.precision
            g = g + prior.precision @ (x - prior.mu)
        step = np.linalg.solve(H, g)
        x = x - step
        if np.linalg.norm(step) < tol:
            break
    r, s, c = terms(x)
    J = np.atleast_2d(jac(x)) / s[:, None]
    H = J.T @ J + (prior.precision if prior is not None else 0)
    F = c + 0.5 * np.linalg.slogdet(H)[1] - 0.5 * n * np.log(2 * np.pi)
    return LaplaceResult(Gaussian(x, np.linalg.inv(H)), F, c, it)


def linear_gaussian_update(prior, J, R, y):
    """Exact posterior for y = J x + noise(R), x ~ prior. The Kalman update, in information form."""
    J, R = np.atleast_2d(J), np.atleast_2d(R)
    Lam = prior.precision + J.T @ np.linalg.solve(R, J)
    eta = prior.precision @ prior.mu + J.T @ np.linalg.solve(R, np.atleast_1d(y))
    Sigma = np.linalg.inv(Lam)
    return Gaussian(Sigma @ eta, Sigma)


def sigmoid(l):
    return 1.0 / (1.0 + np.exp(-np.asarray(l, float)))


def logit(p):
    p = np.asarray(p, float)
    return np.log(p) - np.log1p(-p)


def binary_entropy(p):
    """Entropy of a Bernoulli in nats; 0 at p ∈ {0, 1}."""
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log1p(-p))
