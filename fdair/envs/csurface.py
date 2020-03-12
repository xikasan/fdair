# coding: utf-8

import numpy as np
import xtools as xt
import xtools.simulation as xs


class Elevator:

    NORMAL = 0
    SATURATION = 1
    GAIN = 2
    RATE = 3

    def __init__(self, dt, gain, tau, dead_time, dtype=np.float32):
        self.dt = dt
        self.gain = gain
        self.tau = tau
        self.dead_time = dead_time
        self.dead_step = int(self.dead_time / self.dt)

        self.state = 0
        self.buffer = np.zeros(self.dead_step, dtype=dtype)
        self.counter = 0
        self.dtype = dtype

        self.fail_mode = self.NORMAL
        self.fail_value = 0.0

    def reset(self):
        self.state = 0
        self.buffer = self.buffer * 0
        self.counter = 0

    def step(self, command):
        if self.fail_mode == self.GAIN:
            command *= self.fail_value
        command = self.command_delay(command)
        f = lambda x: (self.gain * command - x) / self.tau
        dstate = xs.no_time_rungekutta(f, self.dt, self.state)
        if self.fail_mode == self.RATE:
            dstate = np.clip(dstate, -self.fail_value, self.fail_value)
        self.state = self.state + dstate * self.dt
        if self.fail_mode == self.SATURATION:
            self.state = np.clip(self.state, -self.fail_value, self.fail_value)
        return self.state

    def command_delay(self, command):
        buffered_cmd = self.buffer[self.counter]
        self.counter = (self.counter + 1) % self.dead_step
        self.buffer[self.counter] = command
        return buffered_cmd

    def set_fail_mode(self, mode, value):
        assert mode in [self.NORMAL, self.SATURATION, self.GAIN, self.RATE]
        self.fail_mode = mode
        self.fail_value = value
