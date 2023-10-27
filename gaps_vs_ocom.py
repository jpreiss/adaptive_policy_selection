"""Compares GAPS to the Ideal OGD Update and a different approximation thereof.

The other approximation, OCO-M, is the method from:

    Online Control with Adversarial Disturbances.
    Naman Agarwal, Brian Bullins, Elad Hazan, Sham Kakade, Karan Singh.
    ICML, 2019.

"""

from copy import deepcopy
import multiprocessing
import time

import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_are

from discrete import MPCHorizonSelector
from GAPS import GAPSEstimator
from lineartracking import LinearTracking
from MPCLTI import MPCLTI, MPCLTI_GAPS
from util import contraction_properties, discretized_double_integrator_2d, fastmode


def run_gaps(lti, horizon, controller, T):
    times = []
    for _ in range(T):
        x, context = lti.observe(horizon)
        t0 = time.time()
        u = controller.decide_action(x, context)
        if isinstance(u, tuple):
            u = u[0]
        grad_tuple = lti.step(u)
        controller.update_param(grad_tuple)
        times.append(time.time() - t0)
    cost_traj, state_traj = lti.reset()
    return cost_traj, state_traj, times


def zero_truncation_gradient(lti, horizon, mpc, start_time, steps):
    t = start_time
    #tf = start_time + steps
    assert np.all(lti.init_state == 0)
    lti_trunc = LinearTracking(
        lti.A, lti.B, lti.Q, lti.R, lti.Qf, lti.init_state,
        traj=lti.V[:, t:], ws=lti.W[:, t:], es=lti.E[:, t:]
    )
    estimator = GAPSEstimator(steps)
    for _ in range(steps):
        x, context = lti_trunc.observe(horizon)
        u, dudx, dudtheta = mpc.decide_action(x, context)
        estimator.add_partial_u(dudx, dudtheta)
        grad_tuple = lti_trunc.step(u)
        G = estimator.update(*grad_tuple)
    return G


def run_truncation(lti, horizon, mpc, T, buf_limit, learning_rate):
    param_history = [mpc.param]
    times = []
    for t in range(T):
        x, context = lti.observe(horizon)
        t0 = time.time()
        u, _, _ = mpc.decide_action(x, context)
        lti.step(u)
        start_time = max(0, t - buf_limit + 1)
        steps = min(buf_limit, t - start_time + 1)
        assert start_time + steps == t + 1
        G = zero_truncation_gradient(lti, horizon, mpc, start_time, steps)
        new_param = mpc.param - learning_rate * G
        # the projection step
        mpc.param = np.clip(new_param, 0.0, 1.0)
        param_history.append(mpc.param)
        times.append(time.time() - t0)
    cost_traj, state_traj = lti.reset()
    return cost_traj, state_traj, param_history, times


def main():

    print("-" * 60)
    print("Comparing GAPS vs. other gradient estimators.")

    if fastmode():
        T = 100
    else:
        T = 100000

    # These parameters all affect the costs of the different MPC horizons.
    DT = 0.1
    W_MAG = 1.0 * DT
    E_MAG = 0.25 * W_MAG
    R_SCALE = 1e-2

    # Set up the LQ problem instance: discretized double integrator.
    A, B, Q, R = discretized_double_integrator_2d(DT)
    n, m = B.shape
    P = solve_discrete_are(A, B, Q, R)
    x_0 = np.zeros(4)

    # Tracking trajectory is zero so we can more easily control the
    # signal-to-noise tradeoff in the disturbance predictions.
    target_trajectory = np.zeros((n, T + 1))

    # Generate the true disturbances and the random variables we'll use to
    # generate noisy predictions.
    np.random.seed(0)
    ws = W_MAG * np.random.uniform(-1, 1, size=(n, T))
    es = E_MAG * np.random.uniform(-1, 1, size=(n, T))

    cost_scale = 206.96  # copied manually from running discrete_vs_cts.py.
    Q *= cost_scale
    R *= cost_scale
    P *= cost_scale

    # The complete problem instance has now been determined.
    LTIsys = LinearTracking(A, B, Q, R, Qf=P, init_state=x_0, traj=target_trajectory, ws=ws, es=es)

    # Using the closed form LQR-optimal linear controller to get tighter contraction parameters.
    K = np.linalg.solve(R + B.T @ P @ B, B.T) @ P @ A
    F = A - B @ K
    rho, C = contraction_properties(F)
    print(f"Contraction properties: {rho = :.3}, {C = :.3f}")

    max_horizon = 7
    initial_param = np.zeros(max_horizon)

    print("Running GAPS...")
    gaps_rate = (1.0 - rho) ** (5.0 / 2) / np.sqrt(T)
    gaps_buffer = int(np.log(T) / (2.0 * np.log(1.0 / rho)) + 1)
    print("GAPS buffer length:", gaps_buffer)
    MPC_instance = MPCLTI_GAPS(initial_param=initial_param, buffer_length=gaps_buffer, learning_rate=gaps_rate)
    gaps_cost_history, gaps_whole_trajectory, gaps_times = run_gaps(LTIsys, max_horizon, MPC_instance, T)
    gaps_param_history = np.stack(MPC_instance.param_history)

    print("Running zero-truncation...")
    mpc = MPCLTI(initial_param.copy(), A, B, Q, R)
    trunc_cost_history, trunc_whole_trajectory, trunc_param_history, trunc_times = run_truncation(
        LTIsys, max_horizon, mpc, T, gaps_buffer + 1, gaps_rate)

    print(f"Running oracle (for less time)...")
    T_oracle = min(2000, T)
    mpc = MPCLTI(initial_param.copy(), A, B, Q, R)
    oracle_cost_history, oracle_whole_trajectory, oracle_param_history, oracle_times = run_truncation(
        LTIsys, max_horizon, mpc, T_oracle, T_oracle + 1, gaps_rate)

    # Save results for later plotting.
    for name, params, costs, times in [
        ("GAPS", gaps_param_history, gaps_cost_history, gaps_times),
        ("OCO-M", trunc_param_history, trunc_cost_history, trunc_times),
        ("Ideal OGD", oracle_param_history, oracle_cost_history, oracle_times),
    ]:
        with np.printoptions(precision=3):
            print(f"{name} final trust parameters: {params[-1]}")
        MPC_opt = MPCLTI(params[-1], A, B, Q, R)
        opt_costs, _, _ = run_gaps(LTIsys, max_horizon, MPC_opt, T)
        assert np.all(MPC_opt.param == params[-1])
        np.savez(
            f"data/gaps_vs_ocom_{name}.npz",
            param_history=params,
            opt_cost_history=opt_costs,
            cost_history=costs,
            times=times,
        )


if __name__ == '__main__':
    main()
