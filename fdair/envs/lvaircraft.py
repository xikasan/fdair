# coding: utf-8

import numpy as np
import xtools as xt
from .dynamics import LVDynamics
from .csurface import Elevator


class LVAircraft:

    def __init__(self, dt, sampling_interval):
        self.dt = dt
        self.sampling_interval = sampling_interval
        self.sampling_interval_step = int(np.ceil(xt.round(self.sampling_interval / self.dt, 2)))
        self.dynamics = LVDynamics(dt)
        self.elevator = Elevator(dt, 1.0, 0.1, 0.06)

    def step(self, command):
        for _ in range(self.sampling_interval_step):
            de = self.elevator.step(command)
            state = self.dynamics.step(de)
        return state

    def reset(self):
        self.elevator.reset()
        self.dynamics.reset()
        return self.dynamics.state
