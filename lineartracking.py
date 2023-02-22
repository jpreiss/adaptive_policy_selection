"""Environment for trajectory-tracking with LTI dynamics."""

import numpy as np

class LinearTracking:
    def __init__(self, A, B, Q, R, Qf, init_state, traj, ws, es=None, w_hats=None):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.Qf = Qf
        self.V = traj
        self.init_state = init_state
        self.partial_f_x = []   # partial f_t/x_t
        self.partial_f_u = []   # partial f_t/u_t
        self.partial_g_x = []   # partial g_t/x_t
        self.partial_g_u = []   # partial g_t/u_t
        self.x_history = [init_state]
        self.cost_history = []

        self.n = self.B.shape[0]  # the dimension of the state
        self.m = self.B.shape[1]  # the dimension of the control action
        self.T = self.V.shape[1] - 1  # the total length of the horizon

        self.sys_params = (A, B, Q, R)
        self.state = init_state
        self.time_counter = 0

        assert es is None or w_hats is None
        self.W = ws
        self.E = es
        self.What = w_hats

        # set the time counter to be zero
        self.time_counter = 0

    # observe the current state, the trajectory to track, and the predicted disturbances. k is the maximum pred horizon.
    def observe(self, k):
        t = self.time_counter  # the current time step

        # construct the predicted disturbances sequence
        pred_horizon = min(self.T - t, k)
        if self.E is not None:
            predicted_disturbances = (
                self.W[:, t:t + pred_horizon] +
                np.cumsum(self.E[:, t:t+pred_horizon], axis=1)
            )
        else:
            assert self.What is not None
            predicted_disturbances = self.What[:, t:t + pred_horizon]

        # is this the final time step?
        is_terminal = False
        if t == self.T:
            is_terminal = True

        context = (is_terminal, predicted_disturbances, self.V[:, t:t + pred_horizon + 1], self.sys_params)
        return self.state, context

    # evolve the system for one step using control_action
    # return the available gradients (f_t/x_t, f_t/u_t, g_t/x_t, g_t/u_t)
    def step(self, control_action):
        t = self.time_counter
        stage_cost = (
            (self.state - self.V[:, t]) @ self.Q @ (self.state - self.V[:, t]) +
            control_action @ self.R @ control_action
        )
        self.cost_history.append(stage_cost)
        grads = (2 * (self.state - self.V[:, t]) @ self.Q, 2 * control_action @ self.R, self.A, self.B)
        self.partial_f_x.append(grads[0])
        self.partial_f_u.append(grads[1])
        self.partial_g_x.append(grads[2])
        self.partial_g_u.append(grads[3])
        self.state = self.A @ self.state + self.B @ control_action + self.W[:, t]
        self.x_history.append(self.state)
        self.time_counter += 1
        grad_tuple = None
        if t == 0:
            grad_tuple = (grads[0], grads[1], None, None)
        else:
            grad_tuple = (grads[0], grads[1], self.partial_g_x[-2], self.partial_g_u[-2])
        return grad_tuple

    # reset the system to the initial state, restart the time counter
    # return the total cost incurred on this trajectory
    def reset(self):
        t = self.time_counter
        final_cost = (self.state - self.V[:, t]) @ self.Qf @ (self.state - self.V[:, t])
        self.cost_history.append(final_cost)
        self.state = self.init_state
        self.time_counter = 0
        self.total_cost = 0
        self.partial_f_x = []
        self.partial_f_u = []
        self.partial_g_x = []
        self.partial_g_u = []
        whole_trajectory = np.transpose(np.stack(self.x_history, axis=0))
        self.x_history = []
        cost_history = np.array(self.cost_history)
        self.cost_history = []
        return cost_history, whole_trajectory

