"""Compares GAPS against a FTL-type algorithm for confidence tuning."""

import sys

import numpy as np
from scipy.linalg import solve_discrete_are

from lineartracking import LinearTracking
from MPCLTI import MPCLTI_GAPS
from util import fastmode


class SelfTuningLambdaConfidentControl:
    """Implementation of self-tuning lambda-confident control.

    Follows the paper:

    Robustness and Consistency in Linear Quadratic Control with Untrusted Predictions
    Tongxin Li, Ruixiao Yang, Guannan Qu, Guanya Shi, Chenkai Yu, Adam Wierman, Steven Low
    Proceedings of the ACM on Measurement and Analysis of Computing Systems, Volume 6, Issue 1
    March 2022
    """

    def __init__(self, A, B, Q, R, w_predictions, initial_trust):
        self.A = A
        self.B = B
        P = solve_discrete_are(A, B, Q, R)
        # The left half of the matrix product that forms the standard LQR controller.
        self.half_K = np.linalg.solve(R + B.T @ P @ B, B.T)
        self.H = B @ self.half_K
        self.PA = P @ A

        K = self.half_K @ P @ A
        F = A - B @ K

        # Pre-compute the powers (F.T ** n) @ P for efficiency.
        T = len(w_predictions)
        self.FTnP = [P]
        for i in range(T):
            self.FTnP.append(F.T @ self.FTnP[-1])

        self.w_hats = w_predictions

        # Can only estimate true w's after next state is revealed.
        self.true_ws = []
        self.prev_x = None
        self.prev_u = None

        self.eta_hats = np.full(T, None)
        self.etas = np.full(T, None)

        self.init_lambda = initial_trust
        self.lambda_history = [initial_trust]


    def _update_etas(self, etas, ws):
        """Applies the recursive update rule from Section 4 of the paper."""
        t = len(self.true_ws)
        assert t >= 1
        # Establish base case for the new s we've never seen before.
        etas[t - 1] = np.zeros(ws[0].size)
        # Recursion includes the new s, base case is the empty sum from prev step.
        for s in range(t):
            etas[s] += self.FTnP[t - 1 - s] @ ws[t - 1]


    def step(self, x):
        """Returns the control action u."""

        if self.prev_x is not None:
            # Compute the true w from the previous timestep.
            prev_w = x - (self.A @ self.prev_x + self.B @ self.prev_u)
            self.true_ws.append(prev_w)

        t = len(self.true_ws)

        if t > 0:
            self._update_etas(self.eta_hats, self.w_hats)
            self._update_etas(self.etas, self.true_ws)

        if t < 2:
            lam = self.init_lambda
        else:
            numerator = sum(
                self.etas[s].T @ self.H @ self.eta_hats[s]
                for s in range(t)
            )
            denominator = sum(
                self.eta_hats[s].T @ self.H @ self.eta_hats[s]
                for s in range(t)
            )
            # The inner product in the numerator means this value can possibly be negative.
            lam = np.clip(numerator / denominator, 0, 1)

        T = len(self.w_hats)
        term1 = self.PA @ x
        term2 = lam * sum(
            self.FTnP[tau - t] @ self.w_hats[tau]
            for tau in range(t, T)
        )
        u = -self.half_K @ (term1 + term2)

        self.prev_x = x
        self.prev_u = u
        self.lambda_history.append(lam)

        return u


def main():
    seed = int(sys.argv[1])
    outpath = sys.argv[2]
    np.random.seed(seed)

    # LQR instance.
    A = 2 * np.eye(1)
    B = np.eye(1)
    Q = np.eye(1)
    R = np.eye(1)
    P = solve_discrete_are(A, B, Q, R)
    x0 = np.zeros(1)

    # Constants that should be reported in the paper.
    if fastmode():
        T = 40
    else:
        T = 400
    W_BOUND = 2
    W_SMALL_FACTOR = 0.01
    HERTZ = 0.01 * (10 ** np.random.uniform(0, 1))
    OMEGA = 2 * np.pi * HERTZ
    PHASE = np.random.uniform(0, 2*np.pi)
    INITIAL_BAD_STEPS_RECIP = 4
    INITIAL_LAMBDA = 1.0
    LEARNING_RATE = 4e-1 / (np.trace(P) * W_BOUND * np.sqrt(T))
    PREDICTION_HORIZON = int(np.log(T))
    GRADIENT_BUFFER = int(np.log(T))

    # Real and corrupted disturbances.
    disturbances = np.sin(OMEGA * np.arange(T) + PHASE)[:, None]
    predicted_disturbances = disturbances.copy()
    become_good = T // INITIAL_BAD_STEPS_RECIP
    predicted_disturbances[:become_good, 0] += W_BOUND * np.random.uniform(-1, 1, size=become_good)
    predicted_disturbances[become_good:, 0] += W_SMALL_FACTOR * W_BOUND * np.random.uniform(-1, 1, size=T-become_good)


    #
    # Lambda-confident rollout.
    #
    confident = SelfTuningLambdaConfidentControl(A, B, Q, R, predicted_disturbances, initial_trust=INITIAL_LAMBDA)
    costs_confident = []
    x = x0
    for w in disturbances:
        u = confident.step(x)
        costs_confident.append(x.T @ Q @ x + u.T @ R @ u)
        x = A @ x + B @ u + w
    costs_confident.append(x.T @ P @ x)


    #
    # Ours.
    #
    mpc = MPCLTI_GAPS(np.array([INITIAL_LAMBDA]), GRADIENT_BUFFER, LEARNING_RATE, horizon=PREDICTION_HORIZON)

    # This experiment is just a regulator, but the LinearTracking API expects a target trajectory.
    target_traj = np.zeros((1, T + 1))
    LTI_instance = LinearTracking(
        A, B, Q, R, Qf=P,
        init_state=x0,
        traj=target_traj,
        ws=disturbances.T,
        w_hats=predicted_disturbances.T
    )

    for t in range(T):
        current_state, context = LTI_instance.observe(PREDICTION_HORIZON)
        control_action = mpc.decide_action(current_state, context)
        grad_tuple = LTI_instance.step(control_action)
        mpc.update_param(grad_tuple)

    costs_ours, whole_trajectory = LTI_instance.reset()

    np.savez(
        outpath,
        become_good=become_good,
        lambda_param=confident.lambda_history,
        gaps_param=np.array(mpc.param_history).squeeze(),
        lambda_cost=costs_confident,
        gaps_cost=costs_ours,
    )


if __name__ == '__main__':
    main()
