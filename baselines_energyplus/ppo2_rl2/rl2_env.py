# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Reinforcement Learning Testbed for Power Consumption Optimization
# This project is licensed under the MIT License, see LICENSE

import os
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
from gym import Env, spaces


@dataclass(frozen=True)
class EnergyPlusTask:
    weather_file: str


class EnergyPlusTaskSampler:
    def __init__(self, weather_files: List[str], seed: int = 0):
        if not weather_files:
            raise ValueError("weather_files must be a non-empty list")
        self._weather_files = [os.path.abspath(p) for p in weather_files]
        self._rng = np.random.RandomState(seed)

    def sample(self) -> EnergyPlusTask:
        weather = self._weather_files[int(self._rng.randint(0, len(self._weather_files)))]
        return EnergyPlusTask(weather_file=weather)


class RL2MetaEnv(Env):
    """
    RL^2-style meta-episode wrapper.

    - Keeps previous action/reward/done in observation.
    - Runs K inner episodes per meta-episode without resetting policy state.
    - Samples a new task (e.g., weather) on each meta reset by rebuilding the inner env.
    """

    def __init__(
        self,
        make_env_fn: Callable[[EnergyPlusTask], Env],
        task_sampler: EnergyPlusTaskSampler,
        meta_episodes: int,
        action_dim: int,
        obs_dim: int,
    ):
        self._make_env_fn = make_env_fn
        self._task_sampler = task_sampler
        self._meta_episodes = int(meta_episodes)
        if self._meta_episodes <= 0:
            raise ValueError("meta_episodes must be >= 1")

        self._env: Optional[Env] = None
        self._inner_ep = 0
        self._current_task: Optional[EnergyPlusTask] = None

        self._action_dim = int(action_dim)
        self._obs_dim = int(obs_dim)

        # Observation: [obs, prev_action, prev_reward, prev_done]
        low = np.concatenate(
            [
                np.full(self._obs_dim, -1.0e9, dtype=np.float32),
                np.full(self._action_dim, -1.0, dtype=np.float32),
                np.array([-1.0e9, 0.0], dtype=np.float32),
            ]
        )
        high = np.concatenate(
            [
                np.full(self._obs_dim, 1.0e9, dtype=np.float32),
                np.full(self._action_dim, 1.0, dtype=np.float32),
                np.array([1.0e9, 1.0], dtype=np.float32),
            ]
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._action_dim,),
            dtype=np.float32,
        )

        self._prev_action = np.zeros(self._action_dim, dtype=np.float32)
        self._prev_reward = 0.0
        self._prev_done = 1.0

    def reset(self):
        self._inner_ep = 0
        self._prev_action = np.zeros(self._action_dim, dtype=np.float32)
        self._prev_reward = 0.0
        self._prev_done = 1.0

        self._reset_task()
        obs = self._env.reset()
        return self._augment_obs(obs)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        obs, reward, done, info = self._env.step(action)

        self._prev_action = action
        self._prev_reward = float(reward)
        self._prev_done = 1.0 if done else 0.0

        if done:
            self._inner_ep += 1
            if self._inner_ep < self._meta_episodes:
                info = dict(info)
                info["inner_done"] = True
                obs = self._env.reset()
                done = False
            else:
                info = dict(info)
                info["meta_done"] = True

        return self._augment_obs(obs), reward, done, info

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    # ----------------------------
    # Helpers
    # ----------------------------
    def _reset_task(self):
        if self._env is not None:
            self._env.close()
        self._current_task = self._task_sampler.sample()
        self._env = self._make_env_fn(self._current_task)

    def _augment_obs(self, obs):
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        tail = np.concatenate(
            [
                self._prev_action,
                np.array([self._prev_reward, self._prev_done], dtype=np.float32),
            ]
        )
        return np.concatenate([obs, tail], dtype=np.float32)


def make_energyplus_env_fn(
    *,
    energyplus_file: str,
    model_file: str,
    log_dir: str,
    seed: int,
    env_kwargs: Optional[dict] = None,
) -> Callable[[EnergyPlusTask], Env]:
    from gym_energyplus.envs.energyplus_ma_single_env import EnergyPlusMASingleEnv

    env_kwargs = dict(env_kwargs or {})

    def _make_env(task: EnergyPlusTask) -> Env:
        timestamp = int(time.time() * 1e6)
        run_log_dir = os.path.join(log_dir, f"rl2-{timestamp}")
        os.makedirs(run_log_dir, exist_ok=True)
        return EnergyPlusMASingleEnv(
            energyplus_file=energyplus_file,
            model_file=model_file,
            weather_file=task.weather_file,
            log_dir=run_log_dir,
            verbose=False,
            seed=seed,
            **env_kwargs,
        )

    return _make_env
