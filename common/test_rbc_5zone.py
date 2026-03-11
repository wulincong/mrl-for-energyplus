#!/usr/bin/env python3
import argparse
import os
import re
import tempfile
import time
from dataclasses import dataclass
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gym_energyplus.envs.energyplus_env import EnergyPlusEnv


@dataclass(frozen=True)
class RbcConfig:
    htg_min: float = 10.0
    htg_max: float = 35.0
    clg_min: float = 15.0
    clg_max: float = 40.0
    deadband: float = 1.0


def clamp(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, x)))


def rbc_action(obs: np.ndarray, cfg: RbcConfig) -> np.ndarray:
    """
    obs: [T_outdoor, T_z1, T_z2, T_z3, T_z4, T_z5]
    returns 10D action: [z1_htg, z1_clg, ..., z5_htg, z5_clg]
    """
    zone_temps = obs[1:6]
    actions: List[float] = []
    for tz in zone_temps:
        # Default comfortable setpoints
        htg = 21.0
        clg = 24.0

        # Simple bang-bang around [22, 25]C comfort band
        if tz < 22.0:
            htg = 22.0
            clg = 26.0
        elif tz > 25.0:
            htg = 20.0
            clg = 23.0

        htg = clamp(htg, cfg.htg_min, cfg.htg_max)
        clg = clamp(clg, cfg.clg_min, cfg.clg_max)

        if clg < htg + cfg.deadband:
            clg = clamp(htg + cfg.deadband, cfg.clg_min, cfg.clg_max)
        if htg > clg - cfg.deadband:
            htg = clamp(clg - cfg.deadband, cfg.htg_min, cfg.htg_max)

        actions.extend([htg, clg])

    return np.asarray(actions, dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser(description="RBC smoke test for 5ZoneAirCooled EnergyPlusEnv")
    p.add_argument(
        "--energyplus",
        default=os.environ.get("ENERGYPLUS", "/usr/local/energyplus-9.5.0"),
        help="EnergyPlus executable path",
    )
    p.add_argument(
        "--model",
        default="EnergyPlus/5Zone/5ZoneAirCooled.idf",
        help="IDF model file path",
    )
    p.add_argument(
        "--weather",
        default="EnergyPlus/Model-9-5-0/WeatherData/USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw",
        help="EPW weather file path",
    )
    p.add_argument(
        "--log-dir",
        default=os.path.join("eplog", f"rbc-5zone-{int(time.time())}"),
        help="Log directory",
    )
    p.add_argument("--steps", type=int, default=5000, help="Max steps before stopping (script stops earlier when episode is done)")
    p.add_argument("--seed", type=int, default=0, help="Random seed (for env)")
    p.add_argument("--fast", dest="fast", action="store_true", help="Use a patched 1-day IDF so the episode finishes quickly (default)")
    p.add_argument("--full", dest="fast", action="store_false", help="Use the original IDF (may take long to finish)")
    p.set_defaults(fast=True)
    p.add_argument("--plot", action="store_true", help="Save a Zone1 day plot under log_dir (default)")
    p.add_argument("--no-plot", dest="plot", action="store_false", help="Disable plot output")
    p.set_defaults(plot=True)
    p.add_argument("--plot-zone", type=int, default=1, help="Zone index to plot (1..5)")
    return p.parse_args()


def write_fast_idf(src_idf: str, dst_idf: str) -> None:
    """
    Create a shortened IDF for fast testing:
    - Run only 1 day in the first RunPeriod
    """
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

def _parse_timestep_per_hour_from_idf(idf_path: str) -> int:
    with open(idf_path, "r") as f:
        txt = f.read()
    m = re.search(r"\bTimestep\s*,\s*([0-9]+)\s*;", txt, re.IGNORECASE)
    if not m:
        return 4
    return int(m.group(1))

def save_zone_day_plot(
    log_dir: str,
    timestep_per_hour: int,
    zone_idx: int,
    outdoor_temps: List[float],
    zone_temps: List[float],
    zone_htg_sps: List[float],
    zone_clg_sps: List[float],
    zone_cool_rates: List[float],
    zone_heat_rates: List[float],
) -> str:
    dt_hours = 1.0 / float(timestep_per_hour)
    n = min(
        len(outdoor_temps),
        len(zone_temps),
        len(zone_htg_sps),
        len(zone_clg_sps),
        len(zone_cool_rates),
        len(zone_heat_rates),
    )
    x = np.arange(n, dtype=np.float64) * dt_hours

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    cool_thr = 850.0  # W
    heat_thr = 850.0  # W
    cool = np.asarray(zone_cool_rates[:n], dtype=np.float64)
    heat = np.asarray(zone_heat_rates[:n], dtype=np.float64)
    total_hvac_power = cool + heat

    mode = np.zeros(n, dtype=np.int8)
    mode[(cool > cool_thr) & (heat <= heat_thr)] = 1  # cooling
    mode[(heat > heat_thr) & (cool <= cool_thr)] = 2  # heating
    mode[(cool > cool_thr) & (heat > heat_thr)] = 3  # both (rare)

    def shade_segments(code: int, color: str, alpha: float, label: str):
        i = 0
        first = True
        while i < n:
            if mode[i] != code:
                i += 1
                continue
            j = i + 1
            while j < n and mode[j] == code:
                j += 1
            ax.axvspan(
                x[i],
                x[j - 1] + dt_hours,
                facecolor=color,
                alpha=alpha,
                linewidth=0,
                label=label if first else None,
            )
            first = False
            i = j

    # Cooling = cold color, Heating = warm color
    shade_segments(1, "#b3d9ff", 0.25, "Cooling On")
    shade_segments(2, "#ffb3b3", 0.25, "Heating On")
    shade_segments(3, "#d1b3ff", 0.20, "Heat+Cool On")

    ax.plot(x, outdoor_temps[:n], label="Outdoor Temp [C]", color="#1f77b4")
    ax.plot(x, zone_temps[:n], label=f"Zone{zone_idx} Temp [C]", color="#ff7f0e")
    ax.step(x, zone_htg_sps[:n], where="post", label=f"Zone{zone_idx} HtgSP [C]", color="#d62728", alpha=0.9)
    ax.step(x, zone_clg_sps[:n], where="post", label=f"Zone{zone_idx} ClgSP [C]", color="#2ca02c", alpha=0.9)

    ax2 = ax.twinx()
    ax2.plot(
        x,
        total_hvac_power,
        label=f"Zone{zone_idx} HVAC Power (Heat+Cool) [W]",
        color="#111111",
        linewidth=1.2,
        alpha=0.9,
    )
    ax2.set_ylabel("HVAC Power [W]")

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Temperature / Setpoint [C]")
    ax.set_xlim(0, max(24.0, float(x[-1]) if len(x) else 24.0))
    ax.grid(True)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, ncol=2, fontsize=9)
    fig.tight_layout()

    out_path = os.path.join(log_dir, f"rbc_zone{zone_idx}_day.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    args = parse_args()

    energyplus = os.path.abspath(args.energyplus)
    model_src = os.path.abspath(args.model)
    weather = os.path.abspath(args.weather)
    log_dir = os.path.abspath(args.log_dir)

    os.makedirs(log_dir, exist_ok=True)
    if args.fast:
        fast_dir = tempfile.mkdtemp(prefix="fast-idf-", dir=log_dir)
        model = os.path.join(fast_dir, "5ZoneAirCooled.idf")
        write_fast_idf(model_src, model)
        print(f"Using fast IDF: {model}")
    else:
        model = model_src

    print(f"EnergyPlus: {energyplus}")
    print(f"Model:      {model}")
    print(f"Weather:    {weather}")
    print(f"Log dir:    {log_dir}")

    timestep_per_hour = _parse_timestep_per_hour_from_idf(model)

    env = EnergyPlusEnv(
        energyplus_file=energyplus,
        model_file=model,
        weather_file=weather,
        log_dir=log_dir,
        verbose=False,
        seed=args.seed,
        framework="ray",  # treat action as raw setpoints (no [-1,1] scaling)
    )

    obs = env.reset()
    obs = np.asarray(obs, dtype=np.float32)
    assert obs.shape[0] >= 6, f"unexpected obs shape: {obs.shape}"

    zone_idx = int(args.plot_zone)
    if zone_idx < 1 or zone_idx > 5:
        raise ValueError("--plot-zone must be in 1..5")
    zone_obs_i = zone_idx  # obs[1] is zone1
    zone_act_i = (zone_idx - 1) * 2

    outdoor_temps = []
    zone_temps = []
    zone_htg_sps = []
    zone_clg_sps = []
    zone_cool_rates = []
    zone_heat_rates = []

    total_reward = 0.0
    cfg = RbcConfig()
    for t in range(args.steps):
        act = rbc_action(obs, cfg)
        assert act.shape == (10,), f"unexpected action shape: {act.shape}"

        outdoor_temps.append(float(obs[0]))
        zone_temps.append(float(obs[zone_obs_i]))
        zone_htg_sps.append(float(act[zone_act_i]))
        zone_clg_sps.append(float(act[zone_act_i + 1]))
        # obs: [Tout, Tz1..Tz5, CoolRate_z1..z5, HeatRate_z1..z5]
        cool_base = 1 + 5
        heat_base = 1 + 5 + 5
        zone_cool_rates.append(float(obs[cool_base + (zone_idx - 1)]))
        zone_heat_rates.append(float(obs[heat_base + (zone_idx - 1)]))

        obs, reward, done, _info = env.step(act)
        obs = np.asarray(obs, dtype=np.float32)
        total_reward += float(reward)
        if (t + 1) % 10 == 0 or done:
            temps = ", ".join(f"{x:5.2f}" for x in obs[1:6])
            print(f"t={t+1:04d} reward={reward:8.3f} temps=[{temps}] done={done}")
        if done:
            break

    env.close()
    steps = t + 1

    plot_path = ""
    if args.plot:
        plot_path = save_zone_day_plot(
            log_dir=log_dir,
            timestep_per_hour=timestep_per_hour,
            zone_idx=zone_idx,
            outdoor_temps=outdoor_temps,
            zone_temps=zone_temps,
            zone_htg_sps=zone_htg_sps,
            zone_clg_sps=zone_clg_sps,
            zone_cool_rates=zone_cool_rates,
            zone_heat_rates=zone_heat_rates,
        )
        print(f"Saved plot: {plot_path}")

    print(f"OK: steps={steps} total_reward={total_reward:.3f} log_dir={log_dir}")


if __name__ == "__main__":
    main()
