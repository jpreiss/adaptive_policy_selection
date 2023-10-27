import os

import numpy as np


def fastmode():
    fast = os.getenv("FAST")
    return fast and fast.lower() == "true"


def contraction_properties(F):
    """Returns contraction properties of F.

    Returns ρ, C such that |F^k| <= C ρ^k, where |.| is the Euclidean operator norm.
    """
    n = F.shape[0]
    rho = np.amax(np.abs(np.linalg.eigvals(F)))
    assert rho < 1

    powers = [np.eye(n)]
    npows = 100
    for _ in range(npows - 1):
        powers.append(F @ powers[-1])
    norms = [np.linalg.norm(Fk, ord=2) for Fk in powers]

    # Make sure we got past the transient phase.
    assert norms[-1] < norms[-2]

    Cs = np.array(norms) / (rho ** np.arange(npows))
    C = np.amax(Cs)
    return rho, C


def discretized_double_integrator_2d(dt):
    R_SCALE = 1e-2

    # Set up the LQ problem instance: discretized double integrator.
    zero2x2 = np.zeros((2, 2))
    A = np.block([
        [np.eye(2), dt * np.eye(2)],
        [  zero2x2,      np.eye(2)],
    ])
    B = np.block([
        [       zero2x2],
        [dt * np.eye(2)],
    ])
    Q = np.block([
        [dt * np.eye(2),              zero2x2],
        [       zero2x2, 0.1 * dt * np.eye(2)],
    ])
    R = R_SCALE * dt * np.eye(2)
    return A, B, Q, R
