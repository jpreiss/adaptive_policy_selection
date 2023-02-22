"""Implements the GAPS gradient estimator."""

import numpy as np


class GAPSEstimator:
    """Only does the gradient estimation - no logging, no projection."""
    def __init__(self, buffer_length):
        self.partial_u_theta = []
        self.partial_u_x = []
        self.buffer_length = buffer_length
        self.buffer = []

    def add_partial_u(self, dudx, dudtheta):
        self.partial_u_x.append(dudx)
        self.partial_u_theta.append(dudtheta)

    def update(self, partial_f_x, partial_f_u, partial_g_x, partial_g_u):
        new_buffer = []
        G = None
        if len(self.buffer) == 0:
            G = partial_f_u @ self.partial_u_theta[-1]
            # update the buffer
            new_buffer = [self.partial_u_theta[-1]]
        else:
            grad_sum = partial_g_u @ self.buffer[0]
            current_buffer_len = len(self.buffer)
            new_buffer = [self.partial_u_theta[-1], partial_g_u @ self.buffer[0]]
            premul = partial_g_x + partial_g_u @ self.partial_u_x[-2]
            if current_buffer_len > 1:
                Brest = self.buffer[1:]
                new = premul @ Brest
                grad_sum += np.sum(new, axis=0)
                new_buffer.extend(list(new))
            G = (partial_f_x + partial_f_u @ self.partial_u_x[-1])@grad_sum + partial_f_u @ self.partial_u_theta[-1]

        if len(new_buffer) > self.buffer_length:
            self.buffer = new_buffer[:self.buffer_length]
        else:
            self.buffer = new_buffer
        return G
