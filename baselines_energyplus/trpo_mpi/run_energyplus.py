#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
import sys

from mpi4py import MPI
from common.energyplus_util import (
    energyplus_arg_parser,
    energyplus_logbase_dir
)
from baselines.common.models import mlp
from baselines.trpo_mpi import trpo_mpi
import os
import datetime
from baselines import logger
from baselines_energyplus.bench import Monitor
import gym
import gym_energyplus  # register EnergyPlus envs
import numpy as np
import re


def make_energyplus_env(env_id, seed):
    """
    Create a wrapped, monitored gym.Env for EnergyEnv
    """
    env = gym.make(env_id)
    env = Monitor(env, logger.get_dir())
    env = EpisodeMetricsWrapper(env, logger.get_dir())
    env.seed(seed)
    return env


class EpisodeMetricsWrapper(gym.Wrapper):
    """
    Log comfort/energy metrics to a CSV alongside monitor.csv.

    Metrics are computed from the flattened 5-zone obs (20 dims):
    [Tout, Tz, CoolRate, HeatRate] * 5
    """

    def __init__(self, env, log_dir):
        super().__init__(env)
        self.log_dir = log_dir
        self.metrics_path = os.path.join(log_dir, "metrics.csv")
        self._reset_counters()
        self._episode_idx = -1
        self._timestep_per_hour = self._parse_timestep_per_hour(os.getenv("ENERGYPLUS_MODEL", ""))
        self._dt_hours = 1.0 / float(self._timestep_per_hour) if self._timestep_per_hour else 0.25

        os.makedirs(log_dir, exist_ok=True)
        self._metrics_f = open(self.metrics_path, "w")
        header = [
            "episode",
            "length",
            "comfort_zone1",
            "comfort_zone2",
            "comfort_zone3",
            "comfort_zone4",
            "comfort_zone5",
            "comfort_mean",
            "energy_zone1_kwh",
            "energy_zone2_kwh",
            "energy_zone3_kwh",
            "energy_zone4_kwh",
            "energy_zone5_kwh",
            "energy_mean_kwh",
        ]
        self._metrics_f.write(",".join(header) + "\n")
        self._metrics_f.flush()

    def reset(self, **kwargs):
        if self._episode_idx >= 0:
            self._flush_episode()
        self._episode_idx += 1
        self._reset_counters()
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._accumulate(obs)
        if done:
            self._flush_episode()
        return obs, rew, done, info

    def close(self):
        try:
            if self._episode_idx >= 0:
                self._flush_episode()
        finally:
            if hasattr(self, "_metrics_f") and self._metrics_f:
                self._metrics_f.flush()
                self._metrics_f.close()
        return self.env.close()

    # ----------------------------
    # Internal
    # ----------------------------
    def _reset_counters(self):
        self._steps = 0
        self._comfort_hits = np.zeros(5, dtype=np.int64)
        self._energy_kwh = np.zeros(5, dtype=np.float64)

    def _accumulate(self, obs):
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        if obs.size != 20:
            # Not the 5-zone flattened obs; skip
            return
        self._steps += 1
        for i in range(5):
            base = i * 4
            tz = obs[base + 1]
            cool_rate = obs[base + 2]
            heat_rate = obs[base + 3]
            if 22.0 <= tz <= 25.0:
                self._comfort_hits[i] += 1
            power_w = max(0.0, cool_rate) + max(0.0, heat_rate)
            self._energy_kwh[i] += power_w * self._dt_hours / 1000.0

    def _flush_episode(self):
        if self._steps == 0:
            return
        comfort = self._comfort_hits.astype(np.float64) / float(self._steps)
        comfort_mean = float(np.mean(comfort))
        energy_mean = float(np.mean(self._energy_kwh))
        row = [
            str(self._episode_idx),
            str(self._steps),
            *[f"{v:.6f}" for v in comfort],
            f"{comfort_mean:.6f}",
            *[f"{v:.6f}" for v in self._energy_kwh],
            f"{energy_mean:.6f}",
        ]
        self._metrics_f.write(",".join(row) + "\n")
        self._metrics_f.flush()

    def _parse_timestep_per_hour(self, idf_path):
        if not idf_path or not os.path.isfile(idf_path):
            return 4
        with open(idf_path, "r") as f:
            txt = f.read()
        m = re.search(r"\bTimestep\s*,\s*([0-9]+)\s*;", txt, re.IGNORECASE)
        if not m:
            return 4
        return int(m.group(1))


def train(env_id, num_timesteps, seed):
    # import baselines.common.tf_util as U
    # sess = U.single_threaded_session()
    # sess.__enter__()
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()

    # Create a new base directory like /tmp/openai-2018-05-21-12-27-22-552435
    log_dir = os.path.join(energyplus_logbase_dir(), datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"))
    if not os.path.exists(log_dir + '/output'):
        os.makedirs(log_dir + '/output')
    os.environ["ENERGYPLUS_LOG"] = log_dir
    model = os.getenv('ENERGYPLUS_MODEL')
    if model is None:
        print('Environment variable ENERGYPLUS_MODEL is not defined')
        sys.exit(1)
    weather = os.getenv('ENERGYPLUS_WEATHER')
    if weather is None:
        print('Environment variable ENERGYPLUS_WEATHER is not defined')
        sys.exit(1)

    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        print('train: init logger with dir={}'.format(log_dir)) #XXX
        logger.configure(log_dir)
    else:
        logger.configure(format_strs=[])
        logger.set_level(logger.DISABLED)

    env = make_energyplus_env(env_id, workerseed)

    trpo_mpi.learn(env=env,
                   network=mlp(num_hidden=32, num_layers=2),
                   total_timesteps=num_timesteps,
                   #timesteps_per_batch=1*1024, max_kl=0.01, cg_iters=10, cg_damping=0.1,
                   timesteps_per_batch=16*1024, max_kl=0.01, cg_iters=10, cg_damping=0.1,
                   gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3)
    env.close()

def main():
    args = energyplus_arg_parser().parse_args()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)

if __name__ == '__main__':
    main()
