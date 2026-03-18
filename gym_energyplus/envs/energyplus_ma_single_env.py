# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Reinforcement Learning Testbed for Power Consumption Optimization
# This project is licensed under the MIT License, see LICENSE

from typing import Dict

import argparse
import os

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
        per_agent_metrics = {}
        for aid, agent_obs in obs_dict.items():
            zone_temp = float(agent_obs[1])
            cool_rate = float(agent_obs[2])
            heat_rate = float(agent_obs[3])
            hvac_power = max(0.0, cool_rate) + max(0.0, heat_rate)
            per_agent_metrics[aid] = {
                "zone_temp": zone_temp,
                "hvac_power": hvac_power,
                "comfort_step_22_25": int(22.0 <= zone_temp <= 25.0),
            }

        info = {
            "per_agent_reward": rew_dict,
            "per_agent_info": info_dict,
            "per_agent_metrics": per_agent_metrics,
        }
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print action/state examples for EnergyPlusMASingleEnv"
    )
    parser.add_argument(
        "--energyplus",
        default=os.environ.get("ENERGYPLUS", "/usr/local/energyplus-9.5.0"),
    )
    parser.add_argument(
        "--model",
        default="EnergyPlus/5Zone/5ZoneAirCooled.idf",
    )
    parser.add_argument(
        "--weather",
        default=(
            "EnergyPlus/Model-9-5-0/WeatherData/"
            "USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw"
        ),
    )
    parser.add_argument("--log_dir", default="eplog/ma-single-demo")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    energyplus_path = os.path.abspath(args.energyplus)
    model_path = os.path.abspath(args.model)
    weather_path = os.path.abspath(args.weather)
    log_dir_path = os.path.abspath(args.log_dir)

    env = EnergyPlusMASingleEnv(
        energyplus_file=energyplus_path,
        model_file=model_path,
        weather_file=weather_path,
        log_dir=log_dir_path,
        seed=args.seed,
        verbose=False,
    )

    try:
        obs = env.reset()
        print("[example] observation_space:", env.observation_space)
        print("[example] action_space:", env.action_space)
        print("[example] reset obs shape:", np.asarray(obs).shape)
        print("[example] reset obs:", np.asarray(obs))

        action = env.action_space.sample()
        print("[example] sampled action shape:", np.asarray(action).shape)
        print("[example] sampled action:", np.asarray(action))

        next_obs, reward, done, info = env.step(action)
        print("[example] step obs shape:", np.asarray(next_obs).shape)
        print("[example] step obs:", np.asarray(next_obs))
        print("[example] reward:", reward)
        print("[example] done:", done)
        print("[example] info keys:", list(info.keys()))
    finally:
        env.close()

'''
[example] observation_space: Box(20,)
[example] action_space: Box(10,)
[example] reset obs shape: (20,)
[example] reset obs: [22.726    22.33748   0.       15.84     22.726    22.334093  0.
  6.84     22.726    22.279612  0.       15.84     22.726    22.462717
  0.        6.84     22.726    22.197456  0.       29.64    ]
[example] sampled action shape: (10,)
[example] sampled action: [-0.50161356 -0.13416038  0.5597785  -0.41787902 -0.40784726  0.44158867
 -0.34240827 -0.4609579  -0.33114165 -0.5443014 ]
ExtCtrlRead: Opened ACT file: /tmp/extctrl_2244883_act
PipeIo.writeline: Opened ACT pipe /tmp/extctrl_2244883_act
[example] step obs shape: (20,)
[example] step obs: [   6.825      21.001638    0.       1141.1235      6.825      21.000225
    0.        480.9806      6.825      21.000692    0.       1129.524
    6.825      20.997995    0.        480.18198     6.825      20.999556
    0.        620.835   ]
[example] reward: -0.16376370264823598
[example] done: False
[example] info keys: ['per_agent_reward', 'per_agent_info', 'per_agent_metrics']
'''