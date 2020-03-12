# coding: utf-8

import numpy as np


class Command:

    def __init__(self, dt, max_amplitude, interval):
        self.dt = dt
        self.max = max_amplitude
        self.interval = int(interval / dt)

        self.prev_target = 0.
        self.next_target = 0.
        self.step_counter = 0
        self.state = 0.

    def step(self):
        self.step_counter += 1
        self.state = self.prev_target + self.step_counter * (self.next_target - self.prev_target) / self.interval
        if self.step_counter == self.interval:
            self.update_target()
        return self.state

    def update_target(self):
        self.prev_target  = self.next_target
        self.next_target  = (np.random.random() * 2 - 1) * self.max
        self.step_counter = 0

    def reset(self, init_state=0.):
        self.state = init_state
        self.next_target = init_state
        self.update_target()
        return self.state


if __name__ == '__main__':
    dt = 0.1
    due = 10
    max_amp = 1.
    interval = 2.
    cmd = Command(dt, max_amp, interval)
    cmd.reset()

    import xtools.simulation as xs
    for time in xs.generate_step_time(due, dt):
        de = cmd.step()
        print(time, de, cmd.next_target)
