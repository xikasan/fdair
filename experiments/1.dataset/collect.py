# coding: utf-8

import numpy as np
import pandas as pd
import xtools as xt
import xtools.simulation as xs
from matplotlib import pyplot as plt
from fdair.envs.lvaircraft import LVAircraft
from fdair.envs.csurface import Elevator
from fdair.utilities.command import Command

MODE = {
    "normal": Elevator.NORMAL,
    "saturation": Elevator.SATURATION,
    "gain": Elevator.GAIN,
    "rate": Elevator.RATE
}


def run(config):
    cf = config if isinstance(config, xt.Config) else xt.Config(config)
    xt.info("config loaded", cf)

    # command
    if cf.command.d2r:
        cf.command.max = xt.d2r(cf.command.max)
    cmd = Command(cf.sampling.dt, cf.command.max, cf.command.interval)
    cmd.reset()
    xt.info("command generator", cmd)

    # aircraft model
    env = LVAircraft(cf.dt, sampling_interval=cf.sampling.dt)
    xt.info("env", env)

    # logger
    capacity = int(cf.due / cf.sampling.dt + 1)
    buf = np.zeros((capacity, cf.sampling.size))
    xt.info("buffer size", buf.shape)
    # save dir
    cf.save.path = xt.join(cf.save.path, cf.fail.mode)
    cf.save.path = xt.make_dirs_current_time(cf.save.path, exist_ok=True)

    # set fail
    cf.fail.mode = MODE[cf.fail.mode]
    cf.fail.value = cf.fail.value if not cf.fail.d2r else xt.d2r(cf.fail.value)
    env.elevator.set_fail_mode(cf.fail.mode, cf.fail.value)

    for round in range(cf.save.num):
        print("-"*60)
        xt.info("round", round)
        roll_over(cf, cmd, env, buf)

        labels = ["time", "dec", "de", "mode", "u", "w", "q", "theta"]
        result = pd.DataFrame({
            key: buf[:, i].flatten() for i, key in enumerate(labels)
        })

        ax = result.plot(x="time", y=["dec", "de"])
        ax.legend(["command", "elevator"])
        # plt.show()
        plt.savefig(xt.join(cf.save.path, "{:03}.png".format(round)))
        plt.close()
        ax.clear()

        result.to_csv(xt.join(cf.save.path, "{:03}.csv".format(round)))
        del result

        buf = reset(env, cmd, buf)


def reset(env, cmd, buf):
    env.reset()
    cmd.reset()
    buf = np.zeros_like(buf)
    return buf


def roll_over(cf, cmd, env, buf):
    i = 0
    dec, state = cmd.state, env.dynamics.state
    store = np.concatenate([[0, dec, env.elevator.state, cf.fail.mode], state])
    buf[i, :] = store

    for time in xs.generate_step_time(cf.due, cf.sampling.dt):
        i += 1
        print(i, time)
        dec = cmd.step()
        state = env.step(dec)
        de = env.elevator.state
        store = np.concatenate([[time, dec, de, cf.fail.mode], state]).astype(np.float32)
        buf[i, :] = store


if __name__ == '__main__':
    xt.go_to_root()
    config_path = "experiments/1.dataset/collect.yaml"

    run(config_path)
