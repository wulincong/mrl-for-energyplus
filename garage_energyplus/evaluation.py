"""Full-year validation for EnergyPlus RL2.

Runs one full-year simulation (all 5 zones simultaneously), then splits
the per-step data into monthly buckets for per-month reporting.
"""
import calendar
import csv
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
from garage.envs import GymEnv
from garage.tf.algos.rl2 import RL2Env

from gym_energyplus.envs.energyplus_multiagent_env import EnergyPlusMultiAgentEnv
from garage_energyplus.env import EplusMonthEnv, ZONE_IDS

# Cumulative day-of-year boundaries for each month (non-leap year)
_MONTH_DAY_ENDS = []
_day = 0
for _m in range(1, 13):
    _day += calendar.monthrange(2013, _m)[1]
    _MONTH_DAY_ENDS.append(_day)  # [31, 59, 90, ..., 365]

STEPS_PER_DAY = 96  # IDF Timestep=4 → 15 min/step → 96 steps/day


def _month_of_step(step_idx: int) -> int:
    """Return month (1-12) for a given 0-based step index."""
    day = step_idx // STEPS_PER_DAY  # 0-based day of year
    for m, end_day in enumerate(_MONTH_DAY_ENDS, start=1):
        if day < end_day:
            return m
    return 12


def evaluate_full_year(
    policy,
    *,
    max_episode_length: int = 35040,
    episodes_per_month: int = 1,  # kept for API compatibility, ignored (always 1 run)
) -> Tuple[List[Dict], Dict]:
    """Evaluate policy on a single full-year EnergyPlus simulation.

    Runs one episode covering the full year for all 5 zones simultaneously,
    then splits per-step data into monthly buckets for per-month reporting.

    Returns:
        yearly_results: list of 12 per-month dicts (r, comfort_ratio, hvac_power)
        summary: aggregated year-level stats
    """
    t_start = time.time()

    # Build a full-year env (original IDF, no month patching)
    run_log_dir = os.path.join(EplusMonthEnv.log_dir, "valid-full-year")
    output_dir = os.path.join(run_log_dir, "output")
    if os.path.isdir(output_dir):
        import shutil
        shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(run_log_dir, exist_ok=True)

    shared_env = EnergyPlusMultiAgentEnv(
        energyplus_file=EplusMonthEnv.energyplus_file,
        model_file=EplusMonthEnv.model_file,
        weather_file=EplusMonthEnv.weather_file,
        log_dir=run_log_dir,
        verbose=False,
        seed=EplusMonthEnv.seed,
        framework="ray",
    )

    # Per-zone, per-month accumulators
    # month_data[zone_idx][month] = {"return": float, "comfort": int, "hvac": float, "steps": int}
    N_ZONES = len(ZONE_IDS)
    month_data = [
        {m: {"return": 0.0, "comfort": 0, "hvac": 0.0, "steps": 0} for m in range(1, 13)}
        for _ in range(N_ZONES)
    ]

    # RL2 augmented obs state per zone
    prev_actions = [np.zeros(2, dtype=np.float32) for _ in range(N_ZONES)]
    prev_rewards = [0.0] * N_ZONES
    prev_dones = [1.0] * N_ZONES  # episode just started

    _reset_policy(policy)
    obs_dict = shared_env.reset()

    zone_obs = []
    for i, zone_id in enumerate(ZONE_IDS):
        raw = np.asarray(obs_dict[zone_id], dtype=np.float32)
        zone_obs.append(_augment_obs(raw, prev_actions[i], prev_rewards[i], prev_dones[i]))

    done = False
    step = 0

    while not done and step < max_episode_length:
        month = _month_of_step(step)
        action_dict = {}
        raw_actions = []

        for i, zone_id in enumerate(ZONE_IDS):
            action, _ = policy.get_action(zone_obs[i])
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            raw_actions.append(action)
            # Scale normalized action to setpoint range
            scaled = _scale_action(action)
            action_dict[zone_id] = scaled

        next_obs_dict, rew_dict, done_dict, _ = shared_env.step(action_dict)
        done = bool(done_dict.get("__all__", False))
        step += 1

        for i, zone_id in enumerate(ZONE_IDS):
            raw_obs = np.asarray(next_obs_dict[zone_id], dtype=np.float32)
            reward = float(rew_dict[zone_id])
            zone_temp = float(raw_obs[1])
            hvac_power = max(0.0, float(raw_obs[2])) + max(0.0, float(raw_obs[3]))

            d = month_data[i][month]
            d["return"] += reward
            d["steps"] += 1
            d["hvac"] += hvac_power
            if 22.0 <= zone_temp <= 25.0:
                d["comfort"] += 1

            is_terminal = done or (step >= max_episode_length)
            zone_obs[i] = _augment_obs(
                raw_obs, raw_actions[i], reward, 1.0 if is_terminal else 0.0
            )
            prev_actions[i] = raw_actions[i]
            prev_rewards[i] = reward
            prev_dones[i] = 1.0 if is_terminal else 0.0

    shared_env.close()

    # Aggregate across zones per month
    yearly_results = []
    all_returns, all_comfort, all_hvac = [], [], []

    for month in range(1, 13):
        zone_returns, zone_comfort, zone_hvac = [], [], []
        for i in range(N_ZONES):
            d = month_data[i][month]
            if d["steps"] == 0:
                continue
            zone_returns.append(d["return"])
            zone_comfort.append(d["comfort"] / d["steps"])
            zone_hvac.append(d["hvac"])

        if not zone_returns:
            continue

        mean_return = float(np.mean(zone_returns))
        mean_comfort = float(np.mean(zone_comfort))
        mean_hvac = float(np.mean(zone_hvac))

        yearly_results.append({
            "month": month,
            "r": round(mean_return, 6),
            "comfort_ratio": round(mean_comfort, 4),
            "hvac_power": round(mean_hvac, 2),
            "l": max_episode_length,
            "t": round(time.time() - t_start, 3),
        })
        all_returns.append(mean_return)
        all_comfort.append(mean_comfort)
        all_hvac.append(mean_hvac)

        print(
            f"  [valid] month={month:2d}  "
            f"return={mean_return:+8.3f}  "
            f"comfort={mean_comfort:.3f}  "
            f"hvac={mean_hvac:.1f}"
        )

    summary = {
        "year_total_return": float(sum(all_returns)),
        "year_total_return_avg": float(np.mean(all_returns)),
        "year_total_return_std": float(np.std(all_returns)),
        "year_comfort_ratio": float(np.mean(all_comfort)),
        "year_hvac_power": float(np.mean(all_hvac)),
    }
    return yearly_results, summary


def save_yearly_validation_csv(
    output_csv: str,
    yearly_results: List[Dict],
    summary: Dict,
    *,
    append: bool = False,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    file_exists = os.path.exists(output_csv)
    mode = "a" if append and file_exists else "w"
    write_header = (not append) or (not file_exists)

    with open(output_csv, mode, newline="") as f:
        if write_header:
            header = {
                "t_start": time.time(),
                "env_id": "EnergyPlusYearValidation",
                **{k: round(v, 6) for k, v in summary.items()},
            }
            f.write(f"#{json.dumps(header)}\n")

        fieldnames = ["month", "r", "comfort_ratio", "hvac_power", "l", "t"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in yearly_results:
            writer.writerow(row)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _reset_policy(policy) -> None:
    if not hasattr(policy, "reset"):
        return
    for args in (([True],), (np.asarray([True]),), ()):
        try:
            policy.reset(*args)
            return
        except TypeError:
            continue


def _augment_obs(obs, prev_action, prev_reward, prev_done):
    return np.concatenate([
        obs.reshape(-1),
        prev_action.reshape(-1),
        np.array([prev_reward, prev_done], dtype=np.float32),
    ]).astype(np.float32)


def _scale_action(action: np.ndarray) -> np.ndarray:
    """Scale normalized [-1,1] action to setpoint range [10-35, 15-40]°C."""
    act_low = np.array([10.0, 15.0], dtype=np.float32)
    act_high = np.array([35.0, 40.0], dtype=np.float32)
    action = np.clip(action[:2], -1.0, 1.0)
    return act_low + (action + 1.0) * 0.5 * (act_high - act_low)


def _unpack_step(step_result):
    """Unpack EnvStep or tuple into (obs, reward, done)."""
    if isinstance(step_result, tuple):
        if len(step_result) == 4:
            obs, reward, done, _ = step_result
        elif len(step_result) == 5:
            obs, reward, term, trunc, _ = step_result
            done = bool(term or trunc)
        else:
            raise TypeError(f"Unexpected step tuple length: {len(step_result)}")
        return np.asarray(obs, dtype=np.float32).reshape(-1), reward, done

    if hasattr(step_result, "observation"):
        obs = np.asarray(step_result.observation, dtype=np.float32).reshape(-1)
        reward = float(getattr(step_result, "reward", 0.0))
        done = bool(getattr(step_result, "terminal", False) or getattr(step_result, "done", False))
        return obs, reward, done

    raise TypeError(f"Unsupported step result type: {type(step_result)!r}")
