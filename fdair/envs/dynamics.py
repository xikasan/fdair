# coding: utf-8

import numpy as np
import xtools as xt
import xtools.simulation as xs


class LVDynamics:

    def __init__(self, dt, dtype=np.float32):
        self.dt = dt
        self.A, self.B = self.construct_matrices(dtype)

        self.state = np.zeros(4, dtype=dtype)
        self.dtype = dtype

    @staticmethod
    def construct_matrices(dtype=np.float32):
        A = np.array([
            [-0.02014,   0.01988, -11.27, -9.8],
            [-0.09468,  -0.452,   222.0,  -0.462],
            [-0.000223, -0.01204,  -1.258, 0.0],
            [ 0.0,       0.0,       1.0,   0.0]
        ], dtype=dtype).T
        B = np.array([1.44, -17.9, -1.16, 0], dtype=dtype)
        return A, B

    def step(self, command):
        f = lambda x: x.dot(self.A) + command * self.B
        dstate = xs.no_time_rungekutta(f, self.dt, self.state)
        self.state = self.state + dstate * self.dt
        return self.state

    def reset(self):
        self.state = np.zeros(4, dtype=self.dtype)
