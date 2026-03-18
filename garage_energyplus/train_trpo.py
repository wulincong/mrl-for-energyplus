#!/usr/bin/env python3
"""
Garage TRPO training script for EnergyPlus 5-zone HVAC control.

Usage:
    python train_trpo.py \
        --energyplus_file /path/to/energyplus \
        --model_file /path/to/building.idf \
        --weather_file /path/to/weather.epw \
        --log_dir ./logs \
        --epochs 500 \
        --batch_size 4000

Key garage concepts:
    - GymEnv:       wraps a standard gym.Env for garage
    - GaussianMLPPolicy: continuous policy with diagonal Gaussian output
    - LinearFeatureBaseline: simple baseline V(s) for variance reduction
    - TRPO:         trust region policy optimization
    - Trainer:      orchestrates sampling + optimization loop
"""

import argparse
import os
import sys

import numpy as np
import torch

# ── garage imports ────────────────────────────────────────────────
from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment import deterministic
from garage.torch.algos import TRPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer
from garage.sampler import RaySampler, LocalSampler

# ── your env ──────────────────────────────────────────────────────
from gym_energyplus.envs.energyplus_ma_single_env import EnergyPlusMASingleEnv


# =====================================================================
#  Helper: build the env (called by garage internally for each worker)
# =====================================================================
def make_env(env_kwargs: dict) -> GymEnv:
    """
    Create and wrap the EnergyPlus single-agent env.

    Parameters
    ----------
    env_kwargs : dict
        Keyword arguments forwarded to EnergyPlusMASingleEnv,
        typically energyplus_file, model_file, weather_file, etc.

    Returns
    -------
    GymEnv
        A garage-compatible environment wrapper.
    """
    raw_env = EnergyPlusMASingleEnv(**env_kwargs)
    # max_episode_length must match your IDF RunPeriod setting.
    # E.g., 1 day at 5-min timestep = 288 steps; 1 month ≈ 8640.
    # Adjust this to your actual simulation length.
    return GymEnv(raw_env, max_episode_length=288)


# =====================================================================
#  Main training function (decorated for garage logging)
# =====================================================================
@wrap_experiment(log_dir="./logs", archive_launch_repo=False)
def train_trpo(ctxt=None, env_kwargs=None, seed=1,
               epochs=500, batch_size=4000,
               discount=0.99, hidden_sizes=(64, 64),
               max_kl_step=0.01, use_ray=False):
    """
    Train a TRPO agent on the EnergyPlus 5-zone env.

    Parameters
    ----------
    ctxt : garage.experiment.ExperimentContext
        Provided automatically by @wrap_experiment.
    env_kwargs : dict
        Forwarded to EnergyPlusMASingleEnv.
    seed : int
        Random seed for reproducibility.
    epochs : int
        Number of training epochs (each epoch = 1 policy update).
    batch_size : int
        Total env steps collected per epoch before updating.
    discount : float
        Discount factor gamma.
    hidden_sizes : tuple of int
        MLP hidden layer sizes for policy and value function.
    max_kl_step : float
        Max KL divergence per TRPO update step.
    use_ray : bool
        If True, use RaySampler for parallel rollouts.
    """
    deterministic.set_seed(seed)
    env_kwargs = env_kwargs or {}

    # ── 1. Environment ────────────────────────────────────────────
    env = make_env(env_kwargs)
    print(f"[INFO] Observation space: {env.spec.observation_space}")
    print(f"[INFO] Action space:      {env.spec.action_space}")

    # ── 2. Trainer (manages the training loop) ────────────────────
    trainer = Trainer(ctxt)

    # ── 3. Policy: Gaussian MLP ───────────────────────────────────
    #   Input dim = 20 (4 obs × 5 zones)
    #   Output dim = 10 (2 actions × 5 zones)
    #   Outputs mean + learned log_std → diagonal Gaussian
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=hidden_sizes,
        hidden_nonlinearity=torch.nn.Tanh,
        output_nonlinearity=None,
    )

    # ── 4. Value function (baseline for advantage estimation) ─────
    value_function = GaussianMLPValueFunction(
        env_spec=env.spec,
        hidden_sizes=(64, 32),
        hidden_nonlinearity=torch.nn.Tanh,
        output_nonlinearity=None,
    )

    # ── 5. Sampler (collects trajectories from the env) ───────────
    if use_ray:
        sampler = RaySampler(
            agents=policy,
            envs=env,
            max_episode_length=env.spec.max_episode_length,
        )
    else:
        sampler = LocalSampler(
            agents=policy,
            envs=env,
            max_episode_length=env.spec.max_episode_length,
        )

    # ── 6. Algorithm: TRPO ────────────────────────────────────────
    algo = TRPO(
        env_spec=env.spec,
        policy=policy,
        value_function=value_function,
        sampler=sampler,
        discount=discount,
        max_kl_step=max_kl_step,
        # Center advantages for stability
        center_adv=True,
    )

    # ── 7. Run training ──────────────────────────────────────────
    trainer.setup(algo, env)
    trainer.train(n_epochs=epochs, batch_size=batch_size)

    # ── 8. Save final policy ──────────────────────────────────────
    policy_path = os.path.join(ctxt.snapshot_dir, "final_policy.pt")
    torch.save(policy.state_dict(), policy_path)
    print(f"[INFO] Policy saved to {policy_path}")

    env.close()
    return policy


# =====================================================================
#  CLI entry point
# =====================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train TRPO on EnergyPlus 5-zone HVAC env via garage"
    )

    # ── EnergyPlus paths ──────────────────────────────────────────
    parser.add_argument("--energyplus_file", type=str, required=True,
                        help="Path to the EnergyPlus executable")
    parser.add_argument("--model_file", type=str, required=True,
                        help="Path to the IDF building model file")
    parser.add_argument("--weather_file", type=str, required=True,
                        help="Path to the EPW weather file")

    # ── Training hyperparameters ──────────────────────────────────
    parser.add_argument("--epochs", type=int, default=500,
                        help="Number of training epochs (default: 500)")
    parser.add_argument("--batch_size", type=int, default=4000,
                        help="Steps per epoch (default: 4000)")
    parser.add_argument("--discount", type=float, default=0.99,
                        help="Discount factor gamma (default: 0.99)")
    parser.add_argument("--max_kl", type=float, default=0.01,
                        help="Max KL divergence per update (default: 0.01)")
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 64],
                        help="Hidden layer sizes (default: 64 64)")

    # ── Misc ──────────────────────────────────────────────────────
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed (default: 1)")
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="Logging directory (default: ./logs)")
    parser.add_argument("--use_ray", action="store_true",
                        help="Use RaySampler for parallel rollouts")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    env_kwargs = {
        "energyplus_file": args.energyplus_file,
        "model_file": args.model_file,
        "weather_file": args.weather_file,
    }

    train_trpo(
        env_kwargs=env_kwargs,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        discount=args.discount,
        hidden_sizes=tuple(args.hidden),
        max_kl_step=args.max_kl,
        use_ray=args.use_ray,
    )
