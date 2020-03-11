# coding: utf-8

import xtools as xt
import xtools.simulation as xs
from envs.lvaircraft import LVAircraft


def run():
    dt = 0.02
    sampling_interval = 0.1
    env = LVAircraft(dt, sampling_interval)

    for time in xs.generate_step_time(10, sampling_interval):
        if time == 5.0:
            env.elevator.set_fail_mode(env.elevator.RATE, xt.d2r(0.1))
        state = env.step(xt.d2r(3))
        print(time, xt.r2d(env.elevator.state))


if __name__ == '__main__':
    run()
