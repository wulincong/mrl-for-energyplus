#!/usr/bin/env python3
import argparse
import os
import re
import tempfile
import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gym_energyplus.envs.energyplus_env import EnergyPlusEnv


@dataclass(frozen=True)
class ExperimentConfig:
    htg_default: float = 21.0
    clg_default: float = 24.0


def parse_args():
    p = argparse.ArgumentParser(description="Compare zone5-only HVAC vs no-HVAC (5ZoneAirCooled)")
    p.add_argument(
        "--energyplus",
        default=os.environ.get("ENERGYPLUS", "/usr/local/energyplus-9.5.0"),
        help="EnergyPlus executable path",
    )
    p.add_argument(
        "--model-zone5",
        default="EnergyPlus/5Zone/5ZoneAirCooled_zone5_only.idf",
        help="IDF with only Zone5 HVAC enabled",
    )
    p.add_argument(
        "--model-nohvac",
        default="EnergyPlus/5Zone/5ZoneAirCooled_no_hvac.idf",
        help="IDF with HVAC disabled for all zones",
    )
    p.add_argument(
        "--weather",
        default="EnergyPlus/Model-9-5-0/WeatherData/USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw",
        help="EPW weather file path",
    )
    p.add_argument(
        "--log-dir",
        default=os.path.join("eplog", f"hvac-compare-{int(time.time())}"),
        help="Base log directory",
    )
    p.add_argument("--steps", type=int, default=2000, help="Max steps before stopping")
    p.add_argument("--seed", type=int, default=0, help="Random seed (for env)")
    p.add_argument("--fast", dest="fast", action="store_true", help="Use a patched 1-day IDF (default)")
    p.add_argument("--full", dest="fast", action="store_false", help="Use the original IDF")
    p.set_defaults(fast=True)
    return p.parse_args()


def write_fast_idf(src_idf: str, dst_idf: str) -> None:
    with open(src_idf, "r") as f:
        lines = f.readlines()

    out = []
    in_runperiod = False
    runperiod_seen = False
    for line in lines:
        if line.lstrip().startswith("RunPeriod,") and not runperiod_seen:
            in_runperiod = True
            runperiod_seen = True

        if in_runperiod:
            line = re.sub(r"^\s*([0-9]+)\s*,\s*(!-\s*Begin Month.*)$", r"    1,                       \2", line)
            line = re.sub(r"^\s*([0-9]+)\s*,\s*(!-\s*Begin Day of Month.*)$", r"    1,                       \2", line)
            line = re.sub(r"^\s*([0-9]+)\s*,\s*(!-\s*End Month.*)$", r"    1,                       \2", line)
            line = re.sub(r"^\s*([0-9]+)\s*,\s*(!-\s*End Day of Month.*)$", r"    1,                       \2", line)
            if ";" in line:
                in_runperiod = False

        out.append(line)

    with open(dst_idf, "w") as f:
        f.writelines(out)


def run_episode(
    energyplus: str,
    model: str,
    weather: str,
    log_dir: str,
    steps: int,
    seed: int,
    action_vec: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    env = EnergyPlusEnv(
        energyplus_file=energyplus,
        model_file=model,
        weather_file=weather,
        log_dir=log_dir,
        verbose=False,
        seed=seed,
        framework="ray",
    )

    obs = env.reset()
    obs = np.asarray(obs, dtype=np.float32)

    obs_list = []
    done = False
    for _ in range(steps):
        obs_list.append(obs.copy())
        obs, _rew, done, _info = env.step(action_vec)
        obs = np.asarray(obs, dtype=np.float32)
        if done:
            break

    env.close()
    obs_arr = np.asarray(obs_list, dtype=np.float32)
    temps = obs_arr[:, 1:6] if obs_arr.size else np.zeros((0, 5), dtype=np.float32)
    return obs_arr, temps


def plot_compare(log_dir: str, temps_zone5: np.ndarray, temps_nohvac: np.ndarray):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    zone_labels = [f"Zone{i}" for i in range(1, 6)]
    label_stride = 10

    for i in range(5):
        axes[0].plot(temps_zone5[:, i], label=zone_labels[i])
        axes[1].plot(temps_nohvac[:, i], label=zone_labels[i])

        if temps_zone5.size:
            for t in range(0, temps_zone5.shape[0], label_stride):
                y = temps_zone5[t, i]
                axes[0].plot(t, y, marker="o", markersize=2)
                axes[0].annotate(
                    f"{y:.1f}",
                    (t, y),
                    textcoords="offset points",
                    xytext=(2, 2),
                    fontsize=6,
                )

        if temps_nohvac.size:
            for t in range(0, temps_nohvac.shape[0], label_stride):
                y = temps_nohvac[t, i]
                axes[1].plot(t, y, marker="o", markersize=2)
                axes[1].annotate(
                    f"{y:.1f}",
                    (t, y),
                    textcoords="offset points",
                    xytext=(2, 2),
                    fontsize=6,
                )

    axes[0].set_title("Zone5-only HVAC enabled")
    axes[1].set_title("All-zone HVAC disabled")

    for ax in axes:
        ax.set_ylabel("Zone Temp [C]")
        ax.grid(True)
        ax.legend(ncol=3, fontsize=8)
    axes[1].set_xlabel("Timestep")

    fig.tight_layout()
    out_path = os.path.join(log_dir, "hvac_compare.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    args = parse_args()
    cfg = ExperimentConfig()

    energyplus = os.path.abspath(args.energyplus)
    model_zone5_src = os.path.abspath(args.model_zone5)
    model_nohvac_src = os.path.abspath(args.model_nohvac)
    weather = os.path.abspath(args.weather)
    log_dir = os.path.abspath(args.log_dir)

    os.makedirs(log_dir, exist_ok=True)
    if args.fast:
        fast_dir = tempfile.mkdtemp(prefix="fast-idf-", dir=log_dir)
        model_zone5 = os.path.join(fast_dir, "5ZoneAirCooled_zone5_only.idf")
        model_nohvac = os.path.join(fast_dir, "5ZoneAirCooled_no_hvac.idf")
        write_fast_idf(model_zone5_src, model_zone5)
        write_fast_idf(model_nohvac_src, model_nohvac)
    else:
        model_zone5 = model_zone5_src
        model_nohvac = model_nohvac_src

    action_vec = np.array([cfg.htg_default, cfg.clg_default] * 5, dtype=np.float32)

    zone5_log = os.path.join(log_dir, "zone5_only")
    nohvac_log = os.path.join(log_dir, "no_hvac")
    os.makedirs(zone5_log, exist_ok=True)
    os.makedirs(nohvac_log, exist_ok=True)

    _, temps_zone5 = run_episode(
        energyplus=energyplus,
        model=model_zone5,
        weather=weather,
        log_dir=zone5_log,
        steps=args.steps,
        seed=args.seed,
        action_vec=action_vec,
    )

    _, temps_nohvac = run_episode(
        energyplus=energyplus,
        model=model_nohvac,
        weather=weather,
        log_dir=nohvac_log,
        steps=args.steps,
        seed=args.seed + 1,
        action_vec=action_vec,
    )

    plot_path = plot_compare(log_dir, temps_zone5, temps_nohvac)
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
