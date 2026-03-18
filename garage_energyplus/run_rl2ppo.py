"""RL2PPO training for EnergyPlus 5-zone building control.

PPO variant of RL2: uses clipped surrogate objective instead of TRPO's
trust-region constraint. Generally faster per update and more memory-friendly.

Usage:
    python garage_energyplus/run_rl2ppo.py \
        --model EnergyPlus/5Zone/5ZoneAirCooled.idf \
        --weather EnergyPlus/Model-9-5-0/WeatherData/USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw \
        --n_epochs 30 \
        --episode_per_task 1 \
        --log_dir eplog/garage-rl2-ppo

    # Full-year mode:
    python garage_energyplus/run_rl2ppo.py \
        --model EnergyPlus/5Zone/5ZoneAirCooled.idf \
        --weather EnergyPlus/Model-9-5-0/WeatherData/USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw \
        --full_year_sim \
        --n_epochs 30 \
        --log_dir eplog/garage-rl2-ppo-full-year
"""
import os

import click
from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment import task_sampler
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import RL2PPO
from garage.tf.algos.rl2 import RL2Env
from garage.tf.policies import GaussianGRUPolicy
from garage.trainer import TFTrainer

from garage_energyplus.env import EplusMonthEnv
from garage_energyplus.sampler import EplusSharedSampler
from garage_energyplus.metrics import print_epoch_metrics, print_validation_summary
from garage_energyplus.evaluation import evaluate_full_year, save_yearly_validation_csv


class EplusRL2PPO(RL2PPO):
    """RL2PPO with per-epoch metrics printing and optional full-year validation."""

    def __init__(self, *args, log_dir, validate_full_year=False,
                 validation_episodes_per_month=1, validation_output="yearly_validation.csv",
                 print_metrics_last_n=50, max_episode_length=2880, **kwargs):
        super().__init__(*args, **kwargs)
        self._log_dir = log_dir
        self._validate_full_year = validate_full_year
        self._validation_episodes_per_month = validation_episodes_per_month
        self._validation_output = validation_output
        self._print_metrics_last_n = print_metrics_last_n
        self._max_episode_length = max_episode_length
        self._output_csv = os.path.join(os.path.abspath(log_dir), validation_output)

    def train(self, trainer):
        last_return = None

        for epoch in trainer.step_epochs():
            if trainer.step_itr % self._n_epochs_per_eval == 0:
                if self._meta_evaluator is not None:
                    self._meta_evaluator.evaluate(self)

            print(f"\n{'='*60}")
            print(f"[epoch {epoch}] Collecting samples from EnergyPlus...")

            trainer.step_episode = trainer.obtain_episodes(
                trainer.step_itr,
                env_update=self._task_sampler.sample(self._meta_batch_size),
            )
            last_return = self.train_once(trainer.step_itr, trainer.step_episode)
            trainer.step_itr += 1

            print_epoch_metrics(
                log_dir=self._log_dir,
                epoch_idx=epoch,
                last_n_episodes=self._print_metrics_last_n,
            )

            if self._validate_full_year:
                print(f"\n[epoch {epoch}] Running full-year validation (12 months x 5 zones)...")
                yearly_results, summary = evaluate_full_year(
                    self.policy,
                    episodes_per_month=self._validation_episodes_per_month,
                )
                save_yearly_validation_csv(
                    self._output_csv, yearly_results, summary, append=True
                )
                print_validation_summary(epoch, yearly_results, summary)

        if self._validate_full_year:
            print(f"\n[done] Validation results saved to: {self._output_csv}")

        return last_return


@click.command()
@click.option("--seed", default=1)
@click.option("--max_episode_length", default=None, type=int,
              help="Timesteps per episode. Defaults to 35040 (full year) or 96 (one day).")
@click.option("--n_epochs", default=30, help="Number of training epochs")
@click.option("--episode_per_task", default=1,
              help="Episodes per zone per epoch")
@click.option("--hidden_dim", default=64, help="GRU hidden state dimension")
@click.option("--lr", default=1e-3, type=float, help="PPO learning rate")
@click.option("--lr_clip_range", default=0.2, type=float,
              help="PPO clip range (epsilon). Typical: 0.1-0.3")
@click.option("--max_opt_epochs", default=10, type=int,
              help="Number of optimization epochs per batch (PPO inner loop)")
@click.option(
    "--energyplus",
    default=os.environ.get("ENERGYPLUS", "/usr/local/energyplus-9.5.0"),
)
@click.option("--model", default="EnergyPlus/5Zone/5ZoneAirCooled.idf")
@click.option(
    "--weather",
    default=(
        "EnergyPlus/Model-9-5-0/WeatherData/"
        "USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw"
    ),
)
@click.option("--log_dir", default="eplog/garage-rl2-ppo")
@click.option("--full_year_sim", is_flag=True, default=False,
              help="Run full-year simulation per episode instead of single-month")
@click.option("--validate_full_year", is_flag=True, default=False,
              help="Run full-year validation after each epoch")
@click.option("--validation_episodes_per_month", default=1)
@click.option("--validation_output", default="yearly_validation.csv")
@click.option("--print_metrics_last_n", default=50)
@wrap_experiment
def run_rl2ppo(
    ctxt,
    seed,
    max_episode_length,
    n_epochs,
    episode_per_task,
    hidden_dim,
    lr,
    lr_clip_range,
    max_opt_epochs,
    energyplus,
    model,
    weather,
    log_dir,
    full_year_sim,
    validate_full_year,
    validation_episodes_per_month,
    validation_output,
    print_metrics_last_n,
):
    set_seed(seed)

    # IDF uses Timestep=4 (15-min intervals): 24h × 4 = 96 steps/day
    STEPS_PER_DAY = 96
    FULL_YEAR_STEPS = 365 * STEPS_PER_DAY  # 35040
    if max_episode_length is None:
        max_episode_length = FULL_YEAR_STEPS if full_year_sim else STEPS_PER_DAY

    EplusMonthEnv.configure(
        energyplus_file=energyplus,
        model_file=model,
        weather_file=weather,
        log_dir=log_dir,
        seed=seed,
        full_year=full_year_sim,
    )

    with TFTrainer(snapshot_config=ctxt) as trainer:
        base_env = EplusMonthEnv()
        rl2_env = RL2Env(GymEnv(base_env, max_episode_length=max_episode_length))
        env_spec = rl2_env.spec

        print(f"[setup] obs_space: {env_spec.observation_space}")
        print(f"[setup] action_space: {env_spec.action_space}")
        print(f"[setup] max_episode_length: {env_spec.max_episode_length}")

        policy = GaussianGRUPolicy(
            name="policy",
            hidden_dim=hidden_dim,
            env_spec=env_spec,
            state_include_action=False,
        )
        baseline = LinearFeatureBaseline(env_spec=env_spec)

        sampler = EplusSharedSampler(
            policy=policy,
            env_spec=env_spec,
            n_episodes_per_trial=episode_per_task,
            max_episode_length=max_episode_length,
        )

        meta_batch_size = 5
        tasks = task_sampler.SetTaskSampler(
            EplusMonthEnv,
            wrapper=lambda env, _: RL2Env(
                GymEnv(env, max_episode_length=max_episode_length)
            ),
        )

        algo = EplusRL2PPO(
            meta_batch_size=meta_batch_size,
            task_sampler=tasks,
            env_spec=env_spec,
            policy=policy,
            baseline=baseline,
            sampler=sampler,
            episodes_per_trial=episode_per_task,
            discount=0.99,
            gae_lambda=0.95,
            lr_clip_range=lr_clip_range,
            optimizer_args=dict(
                learning_rate=lr,
                max_optimization_epochs=max_opt_epochs,
                batch_size=32,
            ),
            # EplusRL2PPO-specific args
            log_dir=os.path.abspath(log_dir),
            validate_full_year=validate_full_year,
            validation_episodes_per_month=validation_episodes_per_month,
            validation_output=validation_output,
            print_metrics_last_n=print_metrics_last_n,
            max_episode_length=max_episode_length,
        )

        trainer.setup(algo, rl2_env)

        batch_size = episode_per_task * max_episode_length * meta_batch_size
        mode_str = "full-year" if full_year_sim else f"single-month (max_ep={max_episode_length})"
        print(f"\n[setup] Starting EplusRL2PPO training  [{mode_str}]")
        print(f"  epochs={n_epochs}, meta_batch_size={meta_batch_size} (5 zones)")
        print(f"  episode_per_task={episode_per_task}, max_episode_length={max_episode_length}")
        print(f"  batch_size={batch_size} steps/epoch")
        print(f"  lr={lr}, lr_clip_range={lr_clip_range}, max_opt_epochs={max_opt_epochs}")
        print(f"  log_dir={os.path.abspath(log_dir)}\n")

        trainer.train(n_epochs=n_epochs, batch_size=batch_size)


if __name__ == "__main__":
    run_rl2ppo()
