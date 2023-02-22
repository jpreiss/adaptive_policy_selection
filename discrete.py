"""Selects the MPC horizon using BAPS."""

import numpy as np

from exp3 import exp3
from MPCLTI import MPCLTI


class MPCHorizonSelector:
    def __init__(self, max_horizon, T, batch, learning_rate=None):

        self.batch = batch

        K = max_horizon + 1
        self.MPCs = [
            MPCLTI(np.ones(k), buffer_length=0, learning_rate=0)
            for k in range(K)
        ]

        rng = np.random.default_rng(seed=1)
        self.exp3 = exp3(rng, arms=K, rate=learning_rate)
        self.arm = next(self.exp3)

        self.i_batch = 0
        self.loss_sum = 0
        self.arm_history = []
        self.loss_history = []

        # TODO: Remove hack - to avoid computing gradients when not needed.
        self.learning_rate = 0

    def decide_action(self, state, context):
        if self.i_batch == self.batch:
            self.arm_history.append(self.arm)
            self.loss_history.append(self.loss_sum)
            self.arm = self.exp3.send(self.loss_sum)
            self.i_batch = 0
            self.loss_sum = 0

        is_terminal, predicted_disturbances, V, sys_params = context
        #  Truncate predicted_disturbances to proper horizon - otherwise MPC
        #  instance will be mad.
        k = self.arm
        context_truncated = (is_terminal, predicted_disturbances[:, :k], V, sys_params)
        mpc = self.MPCs[k]
        u = mpc.decide_action(state, context_truncated)
        cost = state.T @ mpc.Q @ state + u.T @ mpc.R @ u

        self.i_batch += 1
        self.loss_sum += cost

        return u

    def update_param(self, grads):
        pass
