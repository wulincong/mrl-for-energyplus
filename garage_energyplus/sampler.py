"""EplusSharedSampler: custom garage Sampler for 5-zone EnergyPlus.

Key design:
- One EnergyPlusMultiAgentEnv per month (one EnergyPlus process).
- 5 zones share that single process; they are NOT separate processes.
- Each rollout drives all 5 zones in lockstep for n_episodes_per_trial episodes.
- Returns an EpisodeBatch with 5 sub-batches (one per zone), each tagged with
  its worker_number as batch_idx so RL2._process_samples can group them.

This replaces LocalSampler for the EnergyPlus use case.
"""
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from garage import EpisodeBatch, StepType
from garage.sampler import Sampler

from gym_energyplus.envs.energyplus_multiagent_env import EnergyPlusMultiAgentEnv
from garage_energyplus.env import EplusMonthEnv, ZONE_IDS


class EplusSharedSampler(Sampler):
    """Sampler that drives one EnergyPlus process for all 5 zones.

    Args:
        policy: The garage policy (GaussianGRUPolicy).
        env_spec: EnvSpec from RL2Env wrapping GymEnv(EplusMonthEnv).
        n_workers (int): Must equal 5 (one per zone).
        n_episodes_per_trial (int): Episodes per meta-batch per zone.
        max_episode_length (int): Max timesteps per episode.
    """

    N_ZONES = 5

    def __init__(
        self,
        policy,
        env_spec,
        *,
        n_episodes_per_trial: int = 2,
        max_episode_length: int = 288,
    ):
        self._policy = policy
        self._env_spec = env_spec
        self._n_episodes_per_trial = n_episodes_per_trial
        self._max_episode_length = max_episode_length

        # Shared EnergyPlus env — rebuilt when month changes
        self._shared_env: Optional[EnergyPlusMultiAgentEnv] = None
        self._active_month: Optional[int] = None

        # Per-zone EplusMonthEnv instances (for action scaling / logging)
        
        self._zone_envs: List[EplusMonthEnv] = [EplusMonthEnv() for _ in range(self.N_ZONES)]
        self._zone_env = EplusMonthEnv()
        for i, env in enumerate(self._zone_envs):
            env._task_zone = ZONE_IDS[i]

        self.total_env_steps = 0

    # ------------------------------------------------------------------
    # Sampler interface
    # ------------------------------------------------------------------
    def obtain_samples(self, itr, num_samples, agent_update, env_update=None):
        """Collect samples from all 5 zones sharing one EnergyPlus process.

        env_update is a list of 5 SetTaskUpdate objects (one per zone).
        We extract the month from the first one and rebuild the shared env
        if the month changed.
        """
        # Apply agent update
        if isinstance(agent_update, (dict, tuple, np.ndarray)):
            self._policy.set_param_values(agent_update)

        # Extract tasks from env_update
        tasks = self._extract_tasks(env_update)
        month = tasks[0]["month"] if tasks else 1
        for i, task in enumerate(tasks):
            self._zone_env.set_task(task)

        # Rebuild shared env if month changed
        self._ensure_shared_env(month)

        # Collect n_episodes_per_trial episodes for all zones
        all_batches = self._rollout_all_zones()

        samples = EpisodeBatch.concatenate(*all_batches)
        self.total_env_steps += int(sum(samples.lengths))
        return samples

    def shutdown_worker(self):
        if self._shared_env is not None:
            self._shared_env.close()
            self._shared_env = None

    def start_worker(self):
        pass  # Nothing to start; env is created lazily

    # ------------------------------------------------------------------
    # Pickle support: exclude unpicklable EnergyPlus env (OS pipes/locks)
    # ------------------------------------------------------------------
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_shared_env"] = None   # drop live EnergyPlus process
        state["_active_month"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    # ------------------------------------------------------------------
    # Core rollout logic
    # ------------------------------------------------------------------
    def _rollout_all_zones(self) -> List[EpisodeBatch]:
        """Run n_episodes_per_trial episodes, driving all 5 zones together."""
        # Per-zone accumulators
        zone_obs: List[List] = [[] for _ in range(self.N_ZONES)]
        zone_actions: List[List] = [[] for _ in range(self.N_ZONES)]
        zone_rewards: List[List] = [[] for _ in range(self.N_ZONES)]
        zone_step_types: List[List] = [[] for _ in range(self.N_ZONES)]
        zone_env_infos: List[defaultdict] = [defaultdict(list) for _ in range(self.N_ZONES)]
        zone_agent_infos: List[defaultdict] = [defaultdict(list) for _ in range(self.N_ZONES)]
        zone_lengths: List[List[int]] = [[] for _ in range(self.N_ZONES)]
        zone_last_obs: List[List] = [[] for _ in range(self.N_ZONES)]

        # Reset policy hidden state once per meta-batch (RL2 convention)
        self._policy.reset()

        for ep_idx in range(self._n_episodes_per_trial):
            # Reset the shared EnergyPlus env
            obs_dict = self._shared_env.reset()

            # Build RL2-augmented initial obs for each zone
            # RL2Env appends [prev_action(2), prev_reward(1), prev_done(1)]
            prev_actions = [np.zeros(2, dtype=np.float32) for _ in range(self.N_ZONES)]
            prev_rewards = [0.0] * self.N_ZONES
            prev_dones = [1.0] * self.N_ZONES  # 1.0 = episode just started

            zone_prev_obs = []
            for i, zone_id in enumerate(ZONE_IDS):
                raw_obs = np.asarray(obs_dict[zone_id], dtype=np.float32)
                aug_obs = self._augment_obs(raw_obs, prev_actions[i], prev_rewards[i], prev_dones[i])
                zone_prev_obs.append(aug_obs)

            ep_lengths = [0] * self.N_ZONES  # reset per episode
            done = False
            step = 0

            while not done and step < self._max_episode_length:
                # Each zone's policy gets its own augmented obs
                action_dict = {}
                zone_raw_actions = []
                zone_agent_info_step = []

                for i, zone_id in enumerate(ZONE_IDS):
                    a, agent_info = self._policy.get_action(zone_prev_obs[i])
                    a = np.asarray(a, dtype=np.float32).reshape(-1)
                    zone_raw_actions.append(a)
                    zone_agent_info_step.append(agent_info)
                    # Scale to setpoint range for EnergyPlus
                    scaled = self._zone_envs[i].scale_action(a)
                    action_dict[zone_id] = scaled

                # Step the shared env with all zone actions
                next_obs_dict, rew_dict, done_dict, info_dict = self._shared_env.step(action_dict)
                done = bool(done_dict.get("__all__", False))
                step += 1

                for i, zone_id in enumerate(ZONE_IDS):
                    raw_obs = np.asarray(next_obs_dict[zone_id], dtype=np.float32)
                    reward = float(rew_dict[zone_id])
                    is_terminal = done or (step >= self._max_episode_length)
                    if step == 1:
                        step_type = StepType.FIRST
                    elif is_terminal:
                        step_type = StepType.TERMINAL
                    else:
                        step_type = StepType.MID

                    # Record in zone accumulators
                    zone_obs[i].append(zone_prev_obs[i])
                    zone_actions[i].append(zone_raw_actions[i])
                    zone_rewards[i].append(reward)
                    zone_step_types[i].append(step_type)
                    for k, v in zone_agent_info_step[i].items():
                        zone_agent_infos[i][k].append(v)

                    # env_info: flatten info_dict for this zone
                    zone_temp = float(raw_obs[1])
                    hvac_power = max(0.0, float(raw_obs[2])) + max(0.0, float(raw_obs[3]))
                    zone_env_infos[i]["zone_temp"].append(zone_temp)
                    zone_env_infos[i]["hvac_power"].append(hvac_power)
                    zone_env_infos[i]["comfort_22_25"].append(int(22.0 <= zone_temp <= 25.0))
                    zone_env_infos[i]["month"].append(self._zone_envs[i]._task_month)

                    ep_lengths[i] += 1

                    # Update RL2 augmented obs
                    aug_obs = self._augment_obs(
                        raw_obs, zone_raw_actions[i], reward, 1.0 if is_terminal else 0.0
                    )
                    zone_prev_obs[i] = aug_obs
                    prev_actions[i] = zone_raw_actions[i]
                    prev_rewards[i] = reward
                    prev_dones[i] = 1.0 if is_terminal else 0.0

                    # Log to EplusMonthEnv for CSV
                    self._zone_envs[i].record_step(reward, zone_temp, hvac_power)

                if done or step >= self._max_episode_length:
                    for i in range(self.N_ZONES):
                        zone_lengths[i].append(ep_lengths[i])
                        zone_last_obs[i].append(zone_prev_obs[i])
                        self._zone_envs[i].finalize_episode()
                    break

        # Build one EpisodeBatch per zone
        batches = []
        for i in range(self.N_ZONES):
            n_steps = sum(zone_lengths[i])
            if n_steps == 0:
                continue

            # Finalize agent_infos
            agent_infos = {}
            for k, v in zone_agent_infos[i].items():
                agent_infos[k] = np.asarray(v)
            # batch_idx: which "worker" (zone index) this belongs to
            agent_infos["batch_idx"] = np.full(n_steps, i, dtype=np.int32)

            env_infos = {k: np.asarray(v) for k, v in zone_env_infos[i].items()}

            batch = EpisodeBatch(
                env_spec=self._env_spec,
                episode_infos={},
                observations=np.asarray(zone_obs[i], dtype=np.float32),
                last_observations=np.asarray(zone_last_obs[i], dtype=np.float32),
                actions=np.asarray(zone_actions[i], dtype=np.float32),
                rewards=np.asarray(zone_rewards[i], dtype=np.float32),
                step_types=np.asarray(zone_step_types[i], dtype=StepType),
                env_infos=env_infos,
                agent_infos=agent_infos,
                lengths=np.asarray(zone_lengths[i], dtype="i"),
            )
            batches.append(batch)

        return batches

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _augment_obs(
        self,
        obs: np.ndarray,
        prev_action: np.ndarray,
        prev_reward: float,
        prev_done: float,
    ) -> np.ndarray:
        """Replicate RL2Env's obs augmentation: [obs, prev_action, prev_reward, prev_done]."""
        return np.concatenate([
            obs.reshape(-1),
            prev_action.reshape(-1),
            np.array([prev_reward, prev_done], dtype=np.float32),
        ]).astype(np.float32)

    def _ensure_shared_env(self, month: int) -> None:
        if EplusMonthEnv.full_year:
            # Full-year mode: build once, never rebuild
            if self._shared_env is not None:
                return
            run_log_dir = os.path.join(EplusMonthEnv.log_dir, "train-full-year")
            model_path = EplusMonthEnv.model_file  # original IDF, no patching
            label = "full-year"
        else:
            if self._active_month == month and self._shared_env is not None:
                return
            run_log_dir = os.path.join(EplusMonthEnv.log_dir, f"train-month-{month:02d}")
            model_path = EplusMonthEnv.write_month_idf(
                EplusMonthEnv.model_file, month, run_log_dir
            )
            label = f"month={month:02d}"

        if self._shared_env is not None:
            self._shared_env.close()
            self._shared_env = None

        # Clean up previous EnergyPlus output to reclaim space
        output_dir = os.path.join(run_log_dir, "output")
        if os.path.isdir(output_dir):
            import shutil as _shutil
            _shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(run_log_dir, exist_ok=True)

        self._shared_env = EnergyPlusMultiAgentEnv(
            energyplus_file=EplusMonthEnv.energyplus_file,
            model_file=model_path,
            weather_file=EplusMonthEnv.weather_file,
            log_dir=run_log_dir,
            verbose=False,
            seed=EplusMonthEnv.seed,
            framework="ray",
        )
        self._active_month = month
        print(f"[EplusSharedSampler] Built env for {label}, log={run_log_dir}")

    def _extract_tasks(self, env_update) -> List[Dict]:
        """Extract task dicts from a list of SetTaskUpdate objects."""
        if env_update is None:
            return [{"month": 1, "zone_id": z} for z in ZONE_IDS]

        tasks = []
        updates = env_update if isinstance(env_update, list) else [env_update]
        for upd in updates:
            task = getattr(upd, "_task", None)
            if task is None:
                task = {"month": 1, "zone_id": ZONE_IDS[len(tasks) % self.N_ZONES]}
            tasks.append(task)

        # Pad to N_ZONES if needed
        while len(tasks) < self.N_ZONES:
            tasks.append(tasks[-1])
        return tasks[:self.N_ZONES]
