# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Reinforcement Learning Testbed for Power Consumption Optimization
# This project is licensed under the MIT License, see LICENSE

from typing import Dict, Tuple

import argparse
import os

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print action/state examples for EnergyPlusMultiAgentEnv"
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
    parser.add_argument("--log_dir", default="eplog/ma-multi-demo")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    energyplus_path = os.path.abspath(args.energyplus)
    model_path = os.path.abspath(args.model)
    weather_path = os.path.abspath(args.weather)
    log_dir_path = os.path.abspath(args.log_dir)

    env = EnergyPlusMultiAgentEnv(
        energyplus_file=energyplus_path,
        model_file=model_path,
        weather_file=weather_path,
        log_dir=log_dir_path,
        seed=args.seed,
        verbose=False,
    )

    try:
        obs_dict = env.reset()
        print("[example] observation_space(per-agent):", env.observation_space)
        print("[example] action_space(per-agent):", env.action_space)
        print("[example] agent_ids:", env.AGENT_IDS)

        print("[example] reset obs sample:")
        for aid in env.AGENT_IDS:
            obs = np.asarray(obs_dict[aid])
            print(f"  {aid}: shape={obs.shape}, obs={obs}")

        action_dict = {aid: env.action_space.sample() for aid in env.AGENT_IDS}
        print("[example] sampled action_dict:")
        for aid in env.AGENT_IDS:
            action = np.asarray(action_dict[aid])
            print(f"  {aid}: shape={action.shape}, action={action}")

        next_obs_dict, rew_dict, done_dict, info_dict = env.step(action_dict)
        print("[example] step next_obs sample:")
        for aid in env.AGENT_IDS:
            next_obs = np.asarray(next_obs_dict[aid])
            print(f"  {aid}: shape={next_obs.shape}, obs={next_obs}")

        print("[example] rew_dict:", rew_dict)
        print("[example] done_dict:", done_dict)
        print("[example] info_dict keys:", list(info_dict.keys()))
    finally:
        env.close()

'''
[example] observation_space(per-agent): Box(4,)
[example] action_space(per-agent): Box(2,)
[example] agent_ids: ('zone_1', 'zone_2', 'zone_3', 'zone_4', 'zone_5')
[example] reset obs sample:
  zone_1: shape=(4,), obs=[22.726   22.33748  0.      15.84   ]
  zone_2: shape=(4,), obs=[22.726    22.334093  0.        6.84    ]
  zone_3: shape=(4,), obs=[22.726    22.279612  0.       15.84    ]
  zone_4: shape=(4,), obs=[22.726    22.462717  0.        6.84    ]
  zone_5: shape=(4,), obs=[22.726    22.197456  0.       29.64    ]
[example] sampled action_dict:
  zone_1: shape=(2,), action=[31.088589 28.034702]
  zone_2: shape=(2,), action=[21.481365 25.179379]
  zone_3: shape=(2,), action=[20.99938 35.44583]
  zone_4: shape=(2,), action=[13.326278 19.870232]
  zone_5: shape=(2,), action=[30.918587 37.283436]
PipeIo.writeline: Opened ACT pipe /tmp/extctrl_2209203_act
ExtCtrlRead: Opened ACT file: /tmp/extctrl_2209203_act
[example] step next_obs sample:
  zone_1: shape=(4,), obs=[   6.825      21.001638    0.       1141.1235  ]
  zone_2: shape=(4,), obs=[  6.825     21.000225   0.       480.9806  ]
  zone_3: shape=(4,), obs=[   6.825      21.000692    0.       1129.524   ]
  zone_4: shape=(4,), obs=[  6.825     20.997995   0.       480.18198 ]
  zone_5: shape=(4,), obs=[  6.825     20.999556   0.       620.835   ]
[example] rew_dict: {'zone_1': -0.16713018356511308, 'zone_2': -0.16082563782812698, 'zone_3': -0.16721296352365972, 'zone_4': -0.16128507800161979, 'zone_5': -0.16236465032266037}
[example] done_dict: {'zone_1': False, 'zone_2': False, 'zone_3': False, 'zone_4': False, 'zone_5': False, '__all__': False}
[example] info_dict keys: ['zone_1', 'zone_2', 'zone_3', 'zone_4', 'zone_5']
EnergyPlusEnv: Severe error(s) occurred. Error count: -1
EnergyPlusEnv: Check contents of /home/wlc/rl-testbed-for-energyplus/eplog/ma-multi-demo/output/episode-00000000-2209203/eplusout.err
'''