# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Reinforcement Learning Testbed for Power Consumption Optimization
# This project is licensed under the MIT License, see LICENSE

from typing import Dict

import numpy as np
from gym import Env, spaces

from gym_energyplus.envs.energyplus_multiagent_env import EnergyPlusMultiAgentEnv


class EnergyPlusMASingleEnv(Env):
    """
    Single-agent wrapper over EnergyPlusMultiAgentEnv for baselines TRPO.

    - Observation: concatenated per-zone obs (zone_1..zone_5), each 4 dims
      [T_outdoor, T_zone, CoolRate_zone, HeatRate_zone]
      -> total obs dim = 20
    - Action: concatenated per-zone actions (zone_1..zone_5), each 2 dims
      [htg_setpoint, clg_setpoint]
      -> total action dim = 10
    - Reward: mean of per-zone rewards

    This wrapper assumes normalized actions in [-1, 1] and scales to setpoints
    before passing them to the multi-agent env.
    """

    ZONES = 5
    AGENT_IDS = tuple([f"zone_{i}" for i in range(1, ZONES + 1)])

    def __init__(self, **kwargs):
        # Force underlying env to accept raw setpoints (no internal scaling)
        kwargs = dict(kwargs)
        kwargs["framework"] = "ray"
        self._env = EnergyPlusMultiAgentEnv(**kwargs)

        # Normalized action space for baselines
        self.action_space = spaces.Box(
            low=np.array([-1.0] * (self.ZONES * 2), dtype=np.float32),
            high=np.array([1.0] * (self.ZONES * 2), dtype=np.float32),
            dtype=np.float32,
        )

        # Concatenated per-zone obs (zone_1..zone_5)
        self.observation_space = spaces.Box(
            low=np.array([-40.0, -20.0, 0.0, 0.0] * self.ZONES, dtype=np.float32),
            high=np.array([60.0, 60.0, 1.0e7, 1.0e7] * self.ZONES, dtype=np.float32),
            dtype=np.float32,
        )

        # Setpoint bounds for scaling
        lows = []
        highs = []
        for _ in range(self.ZONES):
            lows.extend([10.0, 15.0])
            highs.extend([35.0, 40.0])
        self._act_low = np.array(lows, dtype=np.float32)
        self._act_high = np.array(highs, dtype=np.float32)

    def reset(self):
        obs_dict = self._env.reset()
        return self._flatten_obs(obs_dict)

    def step(self, action):
        action = self._normalize_action(action)
        scaled = self._act_low + (action + 1.0) * 0.5 * (self._act_high - self._act_low)

        action_dict = self._split_action(scaled)
        obs_dict, rew_dict, done_dict, info_dict = self._env.step(action_dict)

        obs = self._flatten_obs(obs_dict)
        reward = float(np.mean(list(rew_dict.values())))
        done = bool(done_dict.get("__all__", False))
        info = {"per_agent_reward": rew_dict, "per_agent_info": info_dict}
        return obs, reward, done, info

    def close(self):
        self._env.close()

    # ----------------------------
    # Helpers
    # ----------------------------
    def _flatten_obs(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        obs = []
        for aid in self.AGENT_IDS:
            obs.extend(list(np.asarray(obs_dict[aid], dtype=np.float32)))
        return np.asarray(obs, dtype=np.float32)

    def _split_action(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        action = np.asarray(action, dtype=np.float32)
        act_dict = {}
        for i, aid in enumerate(self.AGENT_IDS):
            idx = i * 2
            act_dict[aid] = np.array([action[idx], action[idx + 1]], dtype=np.float32)
        return act_dict

    def _normalize_action(self, action) -> np.ndarray:
        # Unwrap common nested shapes (e.g., [array([...])])
        if isinstance(action, (list, tuple)) and len(action) == 1:
            action = action[0]
        action = np.asarray(action)
        if action.dtype == object and action.size == 1:
            action = np.asarray(action.item())
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size == 1:
            # Fallback: broadcast scalar to all actions
            action = np.full(self.ZONES * 2, float(action[0]), dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        return action
