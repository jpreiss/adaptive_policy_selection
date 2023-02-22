"""Model-predictive control for LTI system with GAPS confidence tuning."""

from numba import njit
import numpy as np
from scipy import linalg as la

from GAPS import GAPSEstimator


@njit
def _step(multipliers, multipliersA, max_k, param, control_action, predicted_disturbances, V):
    param_dim = multipliers[0].shape[0]
    grads = np.zeros((param_dim, max_k))
    k = predicted_disturbances.shape[0]
    for i in range(k):
        grads[:, i] = multipliers[i] @ predicted_disturbances[i]
        control_action += param[i]*grads[:, i] + multipliersA[i]@(V[i] - V[i+1])
    return grads


class MPCLTI:
    def __init__(self, initial_param, buffer_length, learning_rate, horizon=None):
        self.param = initial_param
        self.learning_rate = learning_rate
        self.param_history = [initial_param]
        self.A = None
        self.B = None
        self.Q = None
        self.R = None
        self.P = None
        self.K = None
        self.Multipliers = []
        self.MultipliersA = []  # Performance optimization
        self.n = None
        self.m = None
        if horizon is None:
            self.max_k = initial_param.shape[0]
        else:
            assert initial_param.size == 1
            self.max_k = horizon
        self.estimator = GAPSEstimator(buffer_length)

    def decide_action(self, state, context):
        is_terminal, predicted_disturbances, V, sys_params = context
        if self.P is None:
            self.A, self.B, self.Q, self.R = sys_params
            self.P = la.solve_discrete_are(self.A, self.B, self.Q, self.R)
            H = np.linalg.inv(self.R + np.transpose(self.B)@self.P@self.B)@np.transpose(self.B)
            self.K = H@self.P@self.A
            F = self.A - self.B@self.K
            temp_mat = - self.P
            for i in range(self.max_k):
                self.Multipliers.append(H@temp_mat)
                self.MultipliersA.append(H@temp_mat@self.A)
                temp_mat = np.transpose(F)@temp_mat
            if self.max_k > 0:
                self.Multipliers = np.stack(self.Multipliers)
                self.MultipliersA = np.stack(self.MultipliersA)
            self.n = self.B.shape[0]
            self.m = self.B.shape[1]
        k = predicted_disturbances.shape[1]
        control_action = - self.K@(state - V[:, 0])
        dudx = -self.K
        if self.param.size > 1:
            # Copy to make contiguous.
            grads = _step(self.Multipliers, self.MultipliersA, self.max_k, self.param, control_action, predicted_disturbances.T.copy(), V.T.copy())
            # Mutates control_action
            dudtheta = grads
        else:
            grad = np.zeros_like(control_action)
            for i in range(k):
                this_grad = self.Multipliers[i]@(predicted_disturbances[:, i])
                control_action += (self.param*this_grad + self.Multipliers[i]@(self.A@V[:, i] - V[:, i+1]))
                grad += this_grad
            dudtheta = grad[:, None]

        self.estimator.add_partial_u(dudx, dudtheta)
        return control_action

    def update_param(self, grads):
        G = self.estimator.update(*grads)
        new_param = self.param - self.learning_rate * G
        # the projection step
        new_param = np.clip(new_param, 0.0, 1.0)
        self.param = new_param
        self.param_history.append(new_param)
        return self.param
