# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Reinforcement Learning Testbed for Power Consumption Optimization
# This project is licensed under the MIT License, see LICENSE

import os
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gym import spaces

from gym_energyplus.envs.energyplus_model import EnergyPlusModel


class EnergyPlusModel5ZoneAirCooled(EnergyPlusModel):
    """
    Model for `EnergyPlus/5Zone/5ZoneAirCooled.idf`.

    ExtCtrl pipe contract (see IDF EMS program):
    - Observations (16):
        1) outdoor drybulb
        2-6) 5 zone mean air temperatures
        7-11) 5 zone sensible cooling rates (Zone Air System Sensible Cooling Rate) [W]
        12-16) 5 zone sensible heating rates (Zone Air System Sensible Heating Rate) [W]
    - Actions (10): for each zone {heating_setpoint, cooling_setpoint}
      Order: [z1_htg, z1_clg, z2_htg, z2_clg, ..., z5_htg, z5_clg]
    """

    ZONES = ["SPACE1-1", "SPACE2-1", "SPACE3-1", "SPACE4-1", "SPACE5-1"]

    def __init__(self, model_file, log_dir, verbose=False):
        super(EnergyPlusModel5ZoneAirCooled, self).__init__(model_file, log_dir, verbose)
        self.reward_low_limit = -10000.0
        self.df = None

    def setup_spaces(self):
        htg_lo, htg_hi = 10.0, 35.0
        clg_lo, clg_hi = 15.0, 40.0

        lows = []
        highs = []
        for _ in range(5):
            lows.extend([htg_lo, clg_lo])
            highs.extend([htg_hi, clg_hi])

        self.action_space = spaces.Box(
            low=np.array(lows, dtype=np.float32),
            high=np.array(highs, dtype=np.float32),
            dtype=np.float32,
        )

        # Raw observation from pipe:
        # [T_outdoor, T_z1..T_z5, CoolRate_z1..z5, HeatRate_z1..z5]
        self.observation_space = spaces.Box(
            low=np.array([-40.0] + [-20.0] * 5 + [0.0] * 10, dtype=np.float32),
            high=np.array([60.0] + [60.0] * 5 + [1.0e7] * 10, dtype=np.float32),
            dtype=np.float32,
        )

    def set_raw_state(self, raw_state):
        if raw_state is not None:
            self.raw_state = raw_state
        else:
            self.raw_state = np.zeros(16, dtype=np.float64)

    def format_state(self, raw_state):
        return np.asarray(raw_state, dtype=np.float32)

    def compute_reward(self):
        rew, _ = self._compute_reward()
        return rew

    def _compute_reward(self, raw_state=None):
        return self.compute_reward_center23_5_gaussian1_0_trapezoid0_1_hvacpow(raw_state)

    def compute_reward_center23_5_gaussian1_0_trapezoid0_1_hvacpow(self, raw_state=None):
        return self.compute_reward_common(
            temperature_center=23.5,
            temperature_tolerance=0.5,
            temperature_gaussian_weight=1.0,
            temperature_gaussian_sharpness=0.5,
            temperature_trapezoid_weight=0.1,
            action_smoothness_weight=0.001,
            hvac_power_weight=1.0 / 100000.0,
            raw_state=raw_state,
        )

    def compute_reward_common(
        self,
        temperature_center=23.5,
        temperature_tolerance=0.5,
        temperature_gaussian_weight=0.0,
        temperature_gaussian_sharpness=1.0,
        temperature_trapezoid_weight=0.0,
        action_smoothness_weight=0.0,
        hvac_power_weight=0.0,
        raw_state=None,
    ):
        """
        2Zone model-inspired reward:
        - Comfort term based on zone temperatures (gaussian + trapezoid)
        - Energy term based on zone sensible heating/cooling rates (as a proxy for HVAC power)
        """
        st = raw_state if raw_state is not None else self.raw_state
        if st is None:
            return 0.0, {}

        temps = np.asarray(st[1:6], dtype=np.float64)
        cool_rates = np.asarray(st[6:11], dtype=np.float64)
        heat_rates = np.asarray(st[11:16], dtype=np.float64)
        hvac_power = float(np.sum(np.clip(cool_rates, 0.0, None) + np.clip(heat_rates, 0.0, None)))

        # Temperature gaussian reward
        rew_temp_gaussian = float(
            np.sum(
                np.exp(-(temps - temperature_center) * (temps - temperature_center) * temperature_gaussian_sharpness)
            )
            * temperature_gaussian_weight
        )

        # Temperature trapezoid penalty outside [center-tol, center+tol]
        phi_low = temperature_center - temperature_tolerance
        phi_high = temperature_center + temperature_tolerance
        below = np.clip(phi_low - temps, 0.0, None)
        above = np.clip(temps - phi_high, 0.0, None)
        rew_temp_trapezoid = -float(np.sum(below + above)) * temperature_trapezoid_weight

        # Energy penalty (proxy)
        rew_hvac_power = -hvac_power * hvac_power_weight

        # Smooth actions to avoid chattering (works for openai and ray frameworks)
        action_delta = np.asarray(self.action, dtype=np.float64) - np.asarray(self.action_prev, dtype=np.float64)
        rew_action_smooth = -float(np.sum(action_delta * action_delta)) * action_smoothness_weight

        rew = rew_temp_gaussian + rew_temp_trapezoid + rew_hvac_power + rew_action_smooth

        if self.verbose:
            tout = float(st[0])
            print(
                "compute_reward: Tout={:7.3f}, Tz(min/mean/max)={:7.3f}/{:7.3f}/{:7.3f}, "
                "HVAC_Power={:10.2f}, Rew={:8.3f}".format(
                    tout,
                    float(np.min(temps)),
                    float(np.mean(temps)),
                    float(np.max(temps)),
                    hvac_power,
                    rew,
                )
            )

        return rew, {
            "rew_temp_gaussian": rew_temp_gaussian,
            "rew_temp_trapezoid": rew_temp_trapezoid,
            "rew_hvac_power": rew_hvac_power,
            "rew_action_smooth": rew_action_smooth,
            "hvac_power": hvac_power,
        }

    # --------------------------------------------------
    # Episode reading / plotting / dumping
    # --------------------------------------------------
    def _resolve_episode_csv(self, ep):
        if isinstance(ep, str):
            return ep
        ep_dir = self.episode_dirs[ep]
        for file in ["eplusout.csv", "eplusout.csv.gz"]:
            file_path = os.path.join(ep_dir, file)
            if os.path.exists(file_path):
                return file_path
        raise FileNotFoundError(f"No eplusout.csv(.gz) found under {ep_dir}")

    def _col(self, df, key, var):
        candidates = [c for c in df.columns if c.startswith(f"{key}:") and var in c]
        if not candidates:
            raise KeyError(f"Missing column for key={key!r} var={var!r}")
        return candidates[0]

    def read_episode(self, ep):
        file_path = self._resolve_episode_csv(ep)
        print(f"read_episode: file={file_path}")

        df = pd.read_csv(file_path).fillna(method="ffill").fillna(method="bfill")
        self.df = df

        epw_files = glob(os.path.join(os.path.dirname(file_path), "USA_??_*.epw"))
        if len(epw_files) == 1:
            self.weather_key = os.path.basename(epw_files[0])[4:6]
        else:
            self.weather_key = "  "

        self.outdoor_temp = df[self._col(df, "Environment", "Site Outdoor Air Drybulb Temperature")]
        self.zone_temps = [
            df[self._col(df, zone, "Zone Air Temperature")]
            if any(c.startswith(f"{zone}:") and "Zone Air Temperature" in c for c in df.columns)
            else df[self._col(df, zone, "Zone Mean Air Temperature")]
            for zone in self.ZONES
        ]

        # Optional setpoints for plotting if present
        self.zone_htg_setpoints = []
        self.zone_clg_setpoints = []
        for zone in self.ZONES:
            htg_cols = [c for c in df.columns if c.startswith(f"{zone}:") and "Zone Thermostat Heating Setpoint Temperature" in c]
            clg_cols = [c for c in df.columns if c.startswith(f"{zone}:") and "Zone Thermostat Cooling Setpoint Temperature" in c]
            self.zone_htg_setpoints.append(df[htg_cols[0]] if htg_cols else None)
            self.zone_clg_setpoints.append(df[clg_cols[0]] if clg_cols else None)

        # Rewards recomputed from the recorded temperatures (comfort-only)
        self.rewards = []
        for vals in zip(self.zone_temps[0], self.zone_temps[1], self.zone_temps[2], self.zone_temps[3], self.zone_temps[4]):
            temps = np.asarray(vals, dtype=np.float64)
            low, high = 22.0, 25.0
            below = np.clip(low - temps, 0.0, None)
            above = np.clip(temps - high, 0.0, None)
            self.rewards.append(-float(np.sum(below + above)))

    def plot_episode(self, ep):
        if self.df is None:
            self.read_episode(ep)

        plt.rcdefaults()
        plt.rcParams["font.size"] = 8
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))

        for zone, series in zip(self.ZONES, self.zone_temps):
            ax.plot(series.values, label=f"{zone} Temp")

        # Plot setpoints if available
        any_setpoint = any(s is not None for s in self.zone_clg_setpoints) or any(s is not None for s in self.zone_htg_setpoints)
        if any_setpoint:
            for zone, htg, clg in zip(self.ZONES, self.zone_htg_setpoints, self.zone_clg_setpoints):
                if htg is not None:
                    ax.plot(htg.values, linestyle="--", alpha=0.6, label=f"{zone} HtgSP")
                if clg is not None:
                    ax.plot(clg.values, linestyle=":", alpha=0.6, label=f"{zone} ClgSP")

        ax.set_title(f"5ZoneAirCooled Episode ({os.path.basename(self._resolve_episode_csv(ep))})")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Temperature [C]")
        ax.grid(True)
        ax.legend(ncol=3, fontsize=7)
        fig.tight_layout()

    def dump_timesteps(self, log_dir="", csv_file="", **kwargs):
        def rolling_mean(data, size, que):
            out = []
            for d in data:
                que.append(d)
                if len(que) > size:
                    que.pop(0)
                out.append(sum(que) / len(que))
            return out

        self.get_episode_list(log_dir=log_dir, csv_file=csv_file)
        print(f"{self.num_episodes} episodes")

        with open("dump_timesteps.csv", mode="w") as f:
            tot_num_rec = 0
            f.write("Sequence,Episode,SequenceInEpisode,Reward,Tout,Tz1,Tz2,Tz3,Tz4,Tz5,RewardAvg1000\n")
            que = []
            for ep in range(self.num_episodes):
                print(f"Episode {ep}")
                self.read_episode(ep)
                rewards_avg = rolling_mean(self.rewards, 1000, que)
                ep_num_rec = 0
                for rew, tout, tz1, tz2, tz3, tz4, tz5, rew_avg in zip(
                    self.rewards,
                    self.outdoor_temp,
                    self.zone_temps[0],
                    self.zone_temps[1],
                    self.zone_temps[2],
                    self.zone_temps[3],
                    self.zone_temps[4],
                    rewards_avg,
                ):
                    f.write(f"{tot_num_rec},{ep},{ep_num_rec},{rew},{tout},{tz1},{tz2},{tz3},{tz4},{tz5},{rew_avg}\n")
                    tot_num_rec += 1
                    ep_num_rec += 1

    def dump_episodes(self, log_dir="", csv_file="", **kwargs):
        self.get_episode_list(log_dir=log_dir, csv_file=csv_file)
        print(f"{self.num_episodes} episodes")

        with open("dump_episodes.dat", mode="w") as f:
            f.write("#Weather AveT1 AveT2 AveT3 AveT4 AveT5 RewAve  Ep\n")
            for ep in range(self.num_episodes):
                print(f"Episode {ep}")
                self.read_episode(ep)
                aves = [float(np.average(t)) for t in self.zone_temps]
                rew_ave = float(np.average(self.rewards)) if self.rewards else 0.0
                f.write(
                    f"\"{self.weather_key}\" "
                    f"{aves[0]:5.2f} {aves[1]:5.2f} {aves[2]:5.2f} {aves[3]:5.2f} {aves[4]:5.2f} "
                    f"{rew_ave:7.3f} {ep:3d}\n"
                )
