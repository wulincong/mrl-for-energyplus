# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Reinforcement Learning Testbed for Power Consumption Optimization
# This project is licensed under the MIT License, see LICENSE

from typing import Dict, Tuple

import numpy as np
from gym import Env, spaces

from gym_energyplus.envs.energyplus_env import EnergyPlusEnv

try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    MultiAgentEnv = Env


class EnergyPlusMultiAgentEnv(MultiAgentEnv):
    """
    Multi-agent wrapper for 5-zone EnergyPlus env.

    Each zone is treated as an independent agent with its own obs/action/reward.
    Under the hood, this still runs one EnergyPlus simulation instance.
    """

    ZONES = 5
    AGENT_IDS = tuple([f"zone_{i}" for i in range(1, ZONES + 1)])

    def __init__(self, **kwargs):
        self._env = EnergyPlusEnv(**kwargs)

        # Per-agent action: [htg_setpoint, clg_setpoint]
        self._action_space = spaces.Box(
            low=np.array([10.0, 15.0], dtype=np.float32),
            high=np.array([35.0, 40.0], dtype=np.float32),
            dtype=np.float32,
        )
        # Per-agent obs: [T_outdoor, T_zone, CoolRate_zone, HeatRate_zone]
        self._observation_space = spaces.Box(
            low=np.array([-40.0, -20.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([60.0, 60.0, 1.0e7, 1.0e7], dtype=np.float32),
            dtype=np.float32,
        )
        self._prev_actions = {aid: np.array([21.0, 24.0], dtype=np.float32) for aid in self.AGENT_IDS}

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def reset(self):
        obs = self._env.reset()
        return self._split_obs(obs)

    def step(self, action_dict: Dict[str, np.ndarray]):
        joint_action = self._merge_actions(action_dict)
        obs, reward, done, info = self._env.step(joint_action)

        obs_dict = self._split_obs(obs)
        rew_dict = self._compute_rewards(obs_dict, action_dict)
        done_dict = {aid: done for aid in self.AGENT_IDS}
        done_dict["__all__"] = done
        info_dict = {aid: dict(info) for aid in self.AGENT_IDS}
        return obs_dict, rew_dict, done_dict, info_dict

    def close(self):
        self._env.close()

    # ----------------------------
    # Helpers
    # ----------------------------
    def _split_obs(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        # obs: [Tout, Tz1..Tz5, CoolRate_z1..z5, HeatRate_z1..z5]
        obs = self._ensure_obs(obs)
        tout = float(obs[0])
        temps = obs[1:1 + self.ZONES]
        cool_rates = obs[1 + self.ZONES:1 + 2 * self.ZONES]
        heat_rates = obs[1 + 2 * self.ZONES:1 + 3 * self.ZONES]

        obs_dict: Dict[str, np.ndarray] = {}
        for i, aid in enumerate(self.AGENT_IDS):
            obs_dict[aid] = np.array(
                [tout, temps[i], cool_rates[i], heat_rates[i]],
                dtype=np.float32
            )
        return obs_dict

    def _ensure_obs(self, obs):
        if obs is None:
            return np.zeros(1 + 3 * self.ZONES, dtype=np.float32)
        obs = np.asarray(obs, dtype=np.float32)
        if obs.size < 1 + 3 * self.ZONES:
            return np.zeros(1 + 3 * self.ZONES, dtype=np.float32)
        return obs

    def _merge_actions(self, action_dict: Dict[str, np.ndarray]) -> np.ndarray:
        actions = []
        for i, aid in enumerate(self.AGENT_IDS):
            act = action_dict.get(aid)
            if act is None:
                act = self._prev_actions[aid]
            act = np.asarray(act, dtype=np.float32)
            self._prev_actions[aid] = act
            actions.extend([float(act[0]), float(act[1])])
        return np.asarray(actions, dtype=np.float32)

    def _compute_rewards(
        self,
        obs_dict: Dict[str, np.ndarray],
        action_dict: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        # 2Zone-like reward per zone: gaussian + trapezoid + hvac power penalty + action smoothness
        temp_center = 23.5
        temp_tol = 0.5
        gauss_weight = 1.0
        gauss_sharp = 0.5
        trap_weight = 0.1
        hvac_weight = 1.0 / 100000.0
        smooth_weight = 0.001

        rew_dict: Dict[str, float] = {}
        for aid in self.AGENT_IDS:
            tout, tz, cool_rate, heat_rate = obs_dict[aid]

            gauss = np.exp(-(tz - temp_center) * (tz - temp_center) * gauss_sharp) * gauss_weight
            phi_low = temp_center - temp_tol
            phi_high = temp_center + temp_tol
            trap = 0.0
            if tz < phi_low:
                trap = -trap_weight * (phi_low - tz)
            elif tz > phi_high:
                trap = -trap_weight * (tz - phi_high)

            hvac_power = max(0.0, cool_rate) + max(0.0, heat_rate)
            power_pen = -hvac_power * hvac_weight

            prev = self._prev_actions[aid]
            act = action_dict.get(aid, prev)
            act = np.asarray(act, dtype=np.float32)
            delta = act - prev
            smooth = -float(np.sum(delta * delta)) * smooth_weight

            rew_dict[aid] = float(gauss + trap + power_pen + smooth)

        return rew_dict
