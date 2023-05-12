"""Compares GAPS to LQR for stabilizing inverted pendulum with mass changes."""

import argparse

import numpy as np
from scipy.linalg import solve_discrete_are
import scipy.signal as sig
import tqdm

from GAPS import GAPSEstimator
from util import fastmode


class InvertedPendulum:
    def __init__(self, m, l):
        self.u_scale = 1.0 / (m * l ** 2)
        self.sin_th_scale = 9.8 / l

    def step(self, xs, us, dt):
        assert len(xs.shape) == 2
        assert len(us.shape) == 2
        assert xs.shape[1] == 2
        assert us.shape[1] == 1
        thetas, theta_dots = xs.T
        theta_ddots = self.u_scale * us[:, 0] + self.sin_th_scale * np.sin(thetas)
        theta_dots2 = theta_dots + dt * theta_ddots
        thetas2 = thetas + dt * theta_dots2
        xs2 = np.stack([thetas2, theta_dots2]).T
        assert xs2.shape == xs.shape
        return xs2

    def jacobians(self, x, u, dt):
        assert x.shape == (2,)
        assert u.shape == (1,)
        dfdx = np.eye(2) + dt * np.array([
            [0, 1],
            [self.sin_th_scale * np.cos(x[0]), 0],
        ])
        dfdu = dt * np.array([[0], [self.u_scale]])
        return dfdx, dfdu


def discretize(A, B, dt):
        n, m = B.shape
        sys_c = (A, B, np.eye(n), np.zeros((n, m)))
        Ad, Bd, Cd, Dd, _ = sig.cont2discrete(sys_c, dt)
        assert np.all(Cd.flat == np.eye(n).flat)
        assert np.all(Dd.flat == np.zeros((n, m)).flat)
        return Ad, Bd


_Q = np.eye(2)
_R = 0.1 * np.eye(1)


def _cost(xs, us):
    xQs = xs @ _Q
    uRs = us @ _R
    return np.sum(xs * xQs, axis=-1) + np.sum(us * uRs, axis=-1)


def pendulum_linearize(m, l):
    g = 9.8
    A = np.array([[0, 1], [g / l, 0]])
    B = np.array([[0], [1 / (m * l**2)]])
    return A, B


def pendulum_gains_lqrd(m, l, dt):
    A, B = pendulum_linearize(m=m, l=l)
    Ad, Bd = discretize(A, B, dt)
    Qd = dt * _Q
    Rd = dt * _R
    Pd = solve_discrete_are(Ad, Bd, Qd, Rd).astype(np.double)
    Kd = np.linalg.solve(Rd + Bd.T @ Pd @ Bd, Bd.T @ Pd @ Ad)
    return Kd.flatten()


def ulprocess(seed, noise, gamma):
    """Ornstein-Uhlenbeck random walk generator.

    When gamma == 0, degenerates into Gaussian noise.
    """
    npr = np.random.default_rng(seed=seed)
    if gamma < 0 or gamma > 1:
        raise ValueError("gamma should be in [0, 1].")
    x = 0.0
    while True:
        x = gamma * x + noise * npr.normal()
        yield x


def main():
    dt = 0.01  # Discretization time interval.
    N = 10     # Number of step-changes in mass.
    # Number of timesteps per step-change in mass.
    if fastmode():
        T = 100
        rate = 1e-1
    else:
        T = 10000
        rate = 3e-2

    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int)
    parser.add_argument("outpath", type=str)
    parser.add_argument("--walk", action="store_true")
    args = parser.parse_args()

    buf_len = 4 * int(1.0 / dt)
    estimator = GAPSEstimator(buffer_length=buf_len)
    theta = pendulum_gains_lqrd(1.0, 1.0, dt)
    prev_dgdx = None
    prev_dgdu = None

    # Always use same masses sequence - aggregation is over randomness of disturbance.
    np.random.seed(100)
    masses = 2 ** np.random.uniform(-1, 1, size=N)

    if args.walk:
        disturbance = ulprocess(seed=args.seed, noise=0.5 * dt, gamma=0.95)
    else:
        disturbance = ulprocess(seed=args.seed, noise=8.0 * dt, gamma=0.0)
    xs = np.zeros((2, 2))

    x_log = []
    mass_log = []
    theta_log_LQ = []
    theta_log_ours = []
    cost_log = []

    for mass in tqdm.tqdm(masses):
        system = InvertedPendulum(m=mass, l=1.0)

        K_LQ = pendulum_gains_lqrd(mass, l=1.0, dt=dt)

        for i in range(T):
            # Get actions.
            us = np.stack([-theta @ xs[0], -K_LQ @ xs[1]]).reshape(2, 1)

            # Log everything.
            x_log.append(xs)
            theta_log_LQ.append(K_LQ)
            theta_log_ours.append(theta)
            cost_log.append(_cost(xs, us))
            mass_log.append(mass)

            # Get controller derivatives.
            dudx = -np.array(theta)[None, :]
            dudtheta = -xs[0][None, :]
            estimator.add_partial_u(dudx, dudtheta)

            # Get system derivatives.
            dgdx, dgdu = system.jacobians(xs[0], us[0], dt)
            dfdx = 2 * _Q @ xs[0]
            dfdu = 2 * _R @ us[0]
            # Gradient sanity check.
            assert np.dot(dfdx, xs[0]) >= 0
            assert np.dot(dfdu, us[0]) >= 0

            # Gradient step.
            G = estimator.update(dfdx, dfdu, prev_dgdx, prev_dgdu)
            theta = theta - rate * G

            prev_dgdx = dgdx
            prev_dgdu = dgdu

            # Dynamics step.
            xs = system.step(xs, us, dt)
            xs[:, 1] += next(disturbance)

    # Save data.
    x_log = np.stack(x_log)
    np.savez(
        args.outpath,
        dt=dt,
        x_log=x_log,
        theta_log_LQ=np.stack(theta_log_LQ),
        theta_log_ours=np.stack(theta_log_ours),
        cost_log=np.stack(cost_log),
        mass_log=np.array(mass_log),
    )


if __name__ == "__main__":
    main()
