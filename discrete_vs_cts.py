"""Compares MPC horizon selection (BAPS) vs. MPC confidence tuning (GAPS)."""

import multiprocessing
import os

import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_are

from discrete import MPCHorizonSelector
from lineartracking import LinearTracking
from MPCLTI import MPCLTI


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


def run(lti, horizon, controller, T):
    for _ in range(T):
        x, context = lti.observe(horizon)
        u = controller.decide_action(x, context)
        grad_tuple = lti.step(u)
        if controller.learning_rate > 0:
            controller.update_param(grad_tuple)
    cost_traj, state_traj = lti.reset()
    return cost_traj, state_traj


def main():

    print("-" * 60)
    print("Comparing discrete vs. continuous MPC trust optimization.")

    if os.getenv("FAST").lower() == "true":
        T = 10000
    else:
        T = 1000000

    # These parameters all affect the costs of the different MPC horizons.
    DT = 0.1
    W_MAG = 1.0 * DT
    E_MAG = 0.25 * W_MAG
    R_SCALE = 1e-2

    # Set up the LQ problem instance: discretized double integrator.
    zero2x2 = np.zeros((2, 2))
    A = np.block([
        [np.eye(2), DT * np.eye(2)],
        [  zero2x2,      np.eye(2)],
    ])
    B = np.block([
        [       zero2x2],
        [DT * np.eye(2)],
    ])
    Q = np.block([
        [DT * np.eye(2),              zero2x2],
        [       zero2x2, 0.1 * DT * np.eye(2)],
    ])
    R = R_SCALE * DT * np.eye(2)
    n, m = B.shape
    P = solve_discrete_are(A, B, Q, R)
    x_0 = np.zeros(4)

    # The MPC horizons we'll be selecting between.
    max_horizon = 7
    horizons = np.arange(max_horizon + 1)
    n_horizons = len(horizons)

    # Tracking trajectory is zero so we can more easily control the
    # signal-to-noise tradeoff in the disturbance predictions.
    target_trajectory = np.zeros((n, T + 1))

    # Generate the true disturbances and the random variables we'll use to
    # generate noisy predictions.
    np.random.seed(0)
    ws = W_MAG * np.random.uniform(-1, 1, size=(n, T))
    es = E_MAG * np.random.uniform(-1, 1, size=(n, T))

    # The complete problem instance has now been determined.
    LTI_instance = LinearTracking(A, B, Q, R, Qf=P, init_state=x_0, traj=target_trajectory, ws=ws, es=es)

    # First, estimate the costs for each horizon.
    print("Computing costs of each MPC horizon with full trust...")
    cost_estimate_batch = T
    args = [
        (LTI_instance, k, MPCLTI(np.ones(k), 0, 0.0), T)
        for k in horizons
    ]
    pool = multiprocessing.Pool(os.cpu_count() - 1)
    outputs = pool.starmap(run, args)
    cost_histories = [out[0] for out in outputs]

    # Rescale the costs for EXP3.
    cost_histories = np.array(cost_histories)
    step_losses = np.sum(cost_histories, axis=1) / cost_estimate_batch
    cost_scale = 1.0 / np.amax(step_losses)
    step_losses *= cost_scale
    cost_histories *= cost_scale  # Used to compute regret later.
    Q *= cost_scale
    R *= cost_scale
    P *= cost_scale
    # Scaling Q and R by the same value doesn't affect the optimal controller 
    LTI_instance = LinearTracking(A, B, Q, R, Qf=P, init_state=x_0, traj=target_trajectory, ws=ws, es=es)

    # TODO: Currently assuming the MPC contraction parameters are no larger
    # than those of the LQR-optimal linear controller - is this true?
    K = np.linalg.solve(R + B.T @ P @ B, B.T) @ P @ A
    F = A - B @ K
    rho, C = contraction_properties(F)
    print(f"Contraction properties: {rho = :.3}, {C = :.3f}")

    # Run EXP3.
    growth = C / (1.0 - rho)
    exp3_batch = int(2 * growth ** (2.0 / 3.0) * (T / (n_horizons * np.log(n_horizons))) ** (1.0 / 3.0))
    print(f"EXP3: batch = {exp3_batch} (total time is {T})")
    assert exp3_batch < T // 2
    exp3_rate = (growth * n_horizons * T ** 2) ** (-1.0 / 3.0) * np.log(n_horizons) ** (2.0 / 3.0)
    print(f"EXP3: rate = {exp3_rate}")
    selector = MPCHorizonSelector(max_horizon, T, batch=exp3_batch, learning_rate=exp3_rate)
    print("Running EXP3-based horizon selection...")
    dis_cost_history, _ = run(LTI_instance, max_horizon, selector, T)

    # Next, run the continuous algorithm.
    print("Running OCO-based trust parameter optimization...")
    initial_param = np.zeros(max_horizon)
    oco_rate = (1.0 - rho) ** (5.0 / 2) / np.sqrt(T)
    oco_buffer = int(np.log(T) / (2.0 * np.log(1.0 / rho)) + 1)
    MPC_instance = MPCLTI(initial_param=initial_param, buffer_length=oco_buffer, learning_rate=oco_rate)
    cts_cost_history, cts_whole_trajectory = run(LTI_instance, max_horizon, MPC_instance, T)
    param_history = np.stack(MPC_instance.param_history)

    # Continuous regret.
    opt_param = param_history[-1]
    with np.printoptions(precision=3):
        print(f"optimal trust parameters: {opt_param}")
    MPC_cts_opt = MPCLTI(initial_param=opt_param, buffer_length=oco_buffer, learning_rate=0.0)
    cts_opt_cost_history, _ = run(LTI_instance, max_horizon, MPC_cts_opt, T)

    np.savez(
        "data/discrete_vs_cts.npz",
        exp3_batch=exp3_batch,
        horizon_cost_histories=cost_histories,
        dis_cost_history=dis_cost_history,
        dis_arm_history=selector.arm_history,
        cts_param_history=param_history,
        cts_opt_cost_history=cts_opt_cost_history,
        cts_cost_history=cts_cost_history,
    )


if __name__ == '__main__':
    main()
