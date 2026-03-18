# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Reinforcement Learning Testbed for Power Consumption Optimization
# This project is licensed under the MIT License, see LICENSE

import argparse
import os
import time
from typing import List

import numpy as np

from baselines import logger
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines_energyplus.ppo2_rl2 import ppo2_lstm

import gym_energyplus  # noqa: F401  # register envs
from baselines_energyplus.bench.monitor import Monitor
from baselines_energyplus.ppo2_rl2.rl2_env import (
    EnergyPlusTaskSampler,
    RL2MetaEnv,
    make_energyplus_env_fn,
)


def _parse_task_weathers(args) -> List[str]:
    if args.task_weather_file:
        with open(args.task_weather_file, "r") as f:
            lines = [ln.strip() for ln in f.readlines()]
        return [ln for ln in lines if ln and not ln.startswith("#")]
    if args.task_weathers:
        return [p.strip() for p in args.task_weathers.split(",") if p.strip()]
    return [args.weather]


def parse_args():
    p = argparse.ArgumentParser(description="RL^2-style PPO2 training for EnergyPlus 5Zone (meta-episodes)")
    p.add_argument("--energyplus", default=os.environ.get("ENERGYPLUS", "/usr/local/energyplus-9.5.0"))
    p.add_argument("--model", default="EnergyPlus/5Zone/5ZoneAirCooled.idf")
    p.add_argument(
        "--weather",
        default="EnergyPlus/Model-9-5-0/WeatherData/USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw",
    )
    p.add_argument("--task-weathers", default="", help="Comma-separated weather files for task sampling")
    p.add_argument("--task-weather-file", default="", help="Text file of weather files (one per line)")
    p.add_argument("--log-dir", default=os.path.join("eplog", f"ppo2-rl2-{int(time.time())}"))
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--total-timesteps", type=int, default=200000)
    p.add_argument("--nsteps", type=int, default=1024)
    p.add_argument("--nminibatches", type=int, default=1)
    p.add_argument("--noptepochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--cliprange", type=float, default=0.2)
    p.add_argument("--log-interval", type=int, default=10)

    p.add_argument("--meta-episodes", type=int, default=3, help="Inner episodes per meta-episode")
    return p.parse_args()


def make_vec_env(args):
    os.makedirs(args.log_dir, exist_ok=True)

    task_weathers = _parse_task_weathers(args)
    sampler = EnergyPlusTaskSampler(task_weathers, seed=args.seed)

    # Base env dimensions (EnergyPlusMA-Single-v0)
    obs_dim = 20
    action_dim = 10

    def _make():
        make_env_fn = make_energyplus_env_fn(
            energyplus_file=os.path.abspath(args.energyplus),
            model_file=os.path.abspath(args.model),
            log_dir=os.path.abspath(args.log_dir),
            seed=args.seed,
            env_kwargs={},
        )
        env = RL2MetaEnv(
            make_env_fn=make_env_fn,
            task_sampler=sampler,
            meta_episodes=args.meta_episodes,
            action_dim=action_dim,
            obs_dim=obs_dim,
        )
        return Monitor(env, os.path.join(args.log_dir, "monitor"))

    return DummyVecEnv([_make])


def main():
    args = parse_args()
    logger.configure(args.log_dir, format_strs=["stdout", "csv"])

    env = make_vec_env(args)
    ppo2_lstm.learn(
        env=env,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        nsteps=args.nsteps,
        ent_coef=args.ent_coef,
        lr=args.lr,
        vf_coef=args.vf_coef,
        max_grad_norm=0.5,
        gamma=args.gamma,
        lam=args.lam,
        log_interval=args.log_interval,
        nminibatches=args.nminibatches,
        noptepochs=args.noptepochs,
        hidden_size=128,
        cliprange=args.cliprange,
    )
    env.close()


if __name__ == "__main__":
    main()
