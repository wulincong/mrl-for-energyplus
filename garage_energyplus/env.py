"""EnergyPlus month-task environment for garage RL2.

Design:
- One EnergyPlus simulation runs 5 zones simultaneously.
- A "task" is a (month, zone_id) pair.
- EplusMonthEnv wraps EnergyPlusMultiAgentEnv and exposes a single-zone
  gym.Env interface (obs dim=4, action dim=2) for garage's GymEnv/RL2Env.
- The shared multi-agent env is managed externally by EplusSharedSampler;
  this class is only used to define the env_spec and for validation rollouts.
"""
import calendar
import csv
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
from gym import Env, spaces

from gym_energyplus.envs.energyplus_multiagent_env import EnergyPlusMultiAgentEnv
from gym_energyplus.envs.energyplus_ma_single_env import EnergyPlusMASingleEnv

ZONE_IDS = tuple(f"zone_{i}" for i in range(1, 6))
MONTHS = tuple(range(1, 13))


class EplusMonthEnv(Env):
    """Single-zone gym.Env view of a 5-zone EnergyPlus simulation.

    Each instance controls exactly one zone of one month's simulation.
    The underlying EnergyPlusMultiAgentEnv is shared across all 5 zones
    of the same month; see EplusSharedSampler for how sharing is managed.

    Task API (for garage SetTaskSampler):
        sample_tasks(n) -> list of {"month": int, "zone_id": str}
        set_task(task)  -> configure this instance for the given task

    Obs:  [T_outdoor, T_zone, CoolRate_zone, HeatRate_zone]  (dim=4)
    Act:  [htg_setpoint, clg_setpoint] normalized to [-1, 1]  (dim=2)
    """
    # Class-level config (set via configure() before instantiation)
    energyplus_file: str = "/usr/local/energyplus-9.5.0"
    model_file: str = "EnergyPlus/5Zone/5ZoneAirCooled.idf"
    weather_file: str = (
        "EnergyPlus/Model-9-5-0/WeatherData/"
        "USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw"
    )
    log_dir: str = "eplog/garage-month-env"
    seed: int = 0
    full_year: bool = False  # when True, run full-year simulation (no month patching)

    num_tasks = len(ZONE_IDS) * len(MONTHS)  # 60 tasks total

    @classmethod
    def configure(
        cls,
        *,
        energyplus_file: str,
        model_file: str,
        weather_file: str,
        log_dir: str,
        seed: int = 0,
        full_year: bool = False,
    ) -> None:
        cls.energyplus_file = os.path.abspath(energyplus_file)
        cls.model_file = os.path.abspath(model_file)
        cls.weather_file = os.path.abspath(weather_file)
        cls.log_dir = os.path.abspath(log_dir)
        cls.seed = int(seed)
        cls.full_year = bool(full_year)


    def __init__(self):
        self._rng = np.random.RandomState(self.seed)
        self._task_month: int = 1
        self._task_zone: str = "zone_1"

        # Shared env reference — injected by EplusSharedSampler, or built lazily for standalone use
        self._shared_env: Optional[EnergyPlusMultiAgentEnv] = None
        self._owns_env: bool = False   # True when this instance built the env itself
        self._active_month: Optional[int] = None

        self._action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._observation_space = spaces.Box(
            low=np.array([-40.0, -20.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([60.0, 60.0, 1.0e7, 1.0e7], dtype=np.float32),
            dtype=np.float32,
        )
        self._act_low = np.array([10.0, 15.0], dtype=np.float32)
        self._act_high = np.array([35.0, 40.0], dtype=np.float32)

        # Episode tracking for CSV logging
        self._episode_return: float = 0.0
        self._episode_steps: int = 0
        self._episode_id: int = 0
        self._comfort_steps: int = 0
        self._hvac_power_sum: float = 0.0

    # ------------------------------------------------------------------
    # Task API
    # ------------------------------------------------------------------
    def sample_tasks(self, n_tasks: int) -> List[Dict]:
        tasks = []
        while len(tasks) < n_tasks:
            month = int(self._rng.choice(MONTHS))
            for zone_id in ZONE_IDS:
                tasks.append({"month": month, "zone_id": zone_id})
                if len(tasks) >= n_tasks:
                    break
        return tasks

    def set_task(self, task) -> None:
        if isinstance(task, dict):
            self._task_month = int(task.get("month", 1))
            zone = str(task.get("zone_id", "zone_1"))
        elif isinstance(task, (list, tuple)) and len(task) >= 2:
            self._task_month = int(task[0])
            zone = str(task[1])
        else:
            self._task_month = int(task)
            zone = "zone_1"
        self._task_zone = zone if zone in ZONE_IDS else "zone_1"

    # ------------------------------------------------------------------
    # gym.Env interface
    # ------------------------------------------------------------------
    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def reset(self):
        """Reset the environment for this zone's task (month + zone_id).

        When called standalone (e.g. during validation), builds its own
        EnergyPlusMultiAgentEnv if one isn't already attached.
        """
        self._episode_return = 0.0
        self._episode_steps = 0
        self._comfort_steps = 0
        self._hvac_power_sum = 0.0
        self._ensure_own_env()
        obs_dict = self._shared_env.reset()
        raw_obs = np.asarray(obs_dict[self._task_zone], dtype=np.float32)
        return raw_obs

    def step(self, action):
        """Step the environment for this zone only.

        Used during standalone validation. Action is normalized [-1,1].
        Returns (obs, reward, done, info) with zone_temp and hvac_power in info.
        """
        self._ensure_own_env()
        scaled = self.scale_action(np.asarray(action, dtype=np.float32))
        action_dict = {self._task_zone: scaled}
        obs_dict, rew_dict, done_dict, _ = self._shared_env.step(action_dict)

        raw_obs = np.asarray(obs_dict[self._task_zone], dtype=np.float32)
        reward = float(rew_dict[self._task_zone])
        done = bool(done_dict.get("__all__", False))

        zone_temp = float(raw_obs[1])
        hvac_power = max(0.0, float(raw_obs[2])) + max(0.0, float(raw_obs[3]))
        info = {"zone_temp": zone_temp, "hvac_power": hvac_power}

        self.record_step(reward, zone_temp, hvac_power)
        if done:
            self.finalize_episode()

        return raw_obs, reward, done, info

    # ------------------------------------------------------------------
    # Pickle support
    # ------------------------------------------------------------------
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_shared_env"] = None
        state["_owns_env"] = False
        state["_active_month"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def close(self):
        if self._shared_env is not None and self._owns_env:
            self._shared_env.close()
        self._shared_env = None
        self._owns_env = False
        self._active_month = None

    # ------------------------------------------------------------------
    # Helpers used by EplusSharedSampler and standalone step()
    # ------------------------------------------------------------------
    def _ensure_own_env(self) -> None:
        """Build a private EnergyPlusMultiAgentEnv if none is attached."""
        if self.full_year:
            # Full-year mode: env is reused regardless of month
            if self._shared_env is not None:
                return
            run_log_dir = os.path.join(self.log_dir, f"valid-full-year-{self._task_zone}")
            model_path = self.model_file  # use original IDF as-is
        else:
            if self._shared_env is not None and self._active_month == self._task_month:
                return
            run_log_dir = os.path.join(
                self.log_dir, f"valid-month-{self._task_month:02d}-{self._task_zone}"
            )
            model_path = self.write_month_idf(self.model_file, self._task_month, run_log_dir)

        if self._shared_env is not None and self._owns_env:
            self._shared_env.close()
            self._shared_env = None

        output_dir = os.path.join(run_log_dir, "output")
        if os.path.isdir(output_dir):
            import shutil as _shutil
            _shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(run_log_dir, exist_ok=True)

        self._shared_env = EnergyPlusMultiAgentEnv(
            energyplus_file=self.energyplus_file,
            model_file=model_path,
            weather_file=self.weather_file,
            log_dir=run_log_dir,
            verbose=False,
            seed=self.seed,
            framework="ray",
        )
        self._owns_env = True
        self._active_month = self._task_month

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """Scale normalized [-1,1] action to setpoint range."""
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size == 1:
            action = np.array([float(action[0]), float(action[0])], dtype=np.float32)
        action = np.clip(action[:2], -1.0, 1.0)
        return self._act_low + (action + 1.0) * 0.5 * (self._act_high - self._act_low)

    def record_step(self, reward: float, zone_temp: float, hvac_power: float) -> None:
        self._episode_return += reward
        self._episode_steps += 1
        if 22.0 <= zone_temp <= 25.0:
            self._comfort_steps += 1
        self._hvac_power_sum += hvac_power

    def finalize_episode(self) -> Dict:
        """Return episode summary and log to CSV."""
        self._episode_id += 1
        comfort_ratio = (
            self._comfort_steps / self._episode_steps
            if self._episode_steps > 0 else 0.0
        )
        summary = {
            "month": self._task_month,
            "zone_id": self._task_zone,
            "episode_id": self._episode_id,
            "episode_steps": self._episode_steps,
            "episode_return": self._episode_return,
            "comfort_steps": self._comfort_steps,
            "comfort_ratio": comfort_ratio,
            "hvac_power_sum": self._hvac_power_sum,
        }
        self._log_episode(summary)
        # Reset accumulators so next episode is independent
        self._episode_return = 0.0
        self._episode_steps = 0
        self._comfort_steps = 0
        self._hvac_power_sum = 0.0
        return summary

    def _log_episode(self, summary: Dict) -> None:
        os.makedirs(self.log_dir, exist_ok=True)
        csv_path = os.path.join(self.log_dir, "episode_metrics.csv")
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary.keys()) + ["timestamp"])
            if write_header:
                writer.writeheader()
            writer.writerow({**summary, "timestamp": int(time.time())})

    # ------------------------------------------------------------------
    # IDF month patching (shared with old code)
    # ------------------------------------------------------------------
    @staticmethod
    def write_month_idf(src_idf: str, month: int, out_dir: str) -> str:
        days_in_month = calendar.monthrange(2013, month)[1]
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
                line = re.sub(
                    r"^\s*([0-9]+)\s*,\s*(!-\s*Begin Month.*)$",
                    f"    {month},                       \\2", line)
                line = re.sub(
                    r"^\s*([0-9]+)\s*,\s*(!-\s*Begin Day of Month.*)$",
                    "    1,                       \\2", line)
                line = re.sub(
                    r"^\s*([0-9]+)\s*,\s*(!-\s*End Month.*)$",
                    f"    {month},                       \\2", line)
                line = re.sub(
                    r"^\s*([0-9]+)\s*,\s*(!-\s*End Day of Month.*)$",
                    f"    {days_in_month},                       \\2", line)
                if ";" in line:
                    in_runperiod = False
            out.append(line)

        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"5ZoneAirCooled-month-{month:02d}.idf")
        with open(out_path, "w") as f:
            f.writelines(out)
        return out_path


class EplusYearEnv(EnergyPlusMASingleEnv):
    """
    Garage-ready single-agent EnergyPlus environment (5-zone, full year).

    Usage
    -----
    # 1. 实例化之前，用 classmethod 配置
    EplusYearEnv.configure(
        energyplus_file="/usr/local/energyplus-9.5.0",
        model_file="EnergyPlus/5Zone/5ZoneAirCooled.idf",
        weather_file="EnergyPlus/WeatherData/USA_CA_SF_TMY3.epw",
        log_dir="eplog/run-01",
        seed=42,
        verbose=False,
        max_episode_steps=8760,
    )

    # 2. 实例化 & 接入 Garage
    env = GymEnv(EplusYearEnv())
    """
    # Class-level config (set via configure() before instantiation)
    energyplus_file: str = "/usr/local/energyplus-9.5.0"
    model_file: str = "EnergyPlus/5Zone/5ZoneAirCooled.idf"
    weather_file: str = (
        "EnergyPlus/Model-9-5-0/WeatherData/"
        "USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw"
    )
    log_dir: str = "eplog/garage-year-env"
    seed: int = 0
    full_year: bool = False  # when True, run full-year simulation (no month patching)
    max_episode_steps = 35040

    @classmethod
    def configure(
        cls,
        *,
        energyplus_file: str,
        model_file: str,
        weather_file: str,
        log_dir: str,
        seed: int = 0,
        full_year: bool = False,
        verbose=False,
        max_episode_steps = 35040
    ) -> None:
        cls.energyplus_file = os.path.abspath(energyplus_file)
        cls.model_file = os.path.abspath(model_file)
        cls.weather_file = os.path.abspath(weather_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cls.log_dir = os.path.abspath(f"{log_dir}-{timestamp}")
        cls.seed = int(seed)
        cls.full_year = bool(full_year)
        cls.max_episode_steps = max_episode_steps

    def __init__(self, **kwargs):
        super().__init__(
            energyplus_file=self.energyplus_file,
            model_file=self.model_file,
            weather_file=self.weather_file,
            log_dir=self.log_dir,
            verbose=False,
            seed=self.seed,
            framework="ray",
        )
        self._step_count = 0

    def reset(self) -> np.ndarray:
        self._step_count = 0
        return super().reset()

    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, done, info = super().step(action)
        self._step_count += 1
        if self._step_count >= self.max_episode_steps:
            done = True
            info["TimeLimit.truncated"] = True
        return obs, reward, done, info

TASK_POOL_V1:List[Dict[str, Any]] = [
    {"seed":0, "tag":f"task_{i}"} for i in range(8)
]

class EplusMetaEnv(EplusYearEnv):

    def __init__(self, task: Dict[str, Any] = None, **kwargs):
        super().__init__(**kwargs)
        self._current_task = task or TASK_POOL_V1[0]

    def sample_tasks(self, num_tasks:int) -> List[Dict[str, Any]]:
        return [TASK_POOL_V1[0]] * num_tasks

    def set_task(self, task: Dict[str, Any])-> None:
        self._current_task = task
        self.seed = 0

    def get_task(self) -> Dict[str, Any]:
        return self._current_task

    def reset(self) -> np.ndarray:
        np.random.seed(0)
        return super().reset()


