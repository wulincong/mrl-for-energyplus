from garage import wrap_experiment
from garage.envs import GymEnv                          # ← 现成的 Garage 适配工具
from garage.experiment.deterministic import set_seed
from garage.torch.algos import PPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer
from garage.sampler import LocalSampler, DefaultWorker
from env import EplusYearEnv

# 统一配置
EplusYearEnv.configure(
    energyplus_file="/usr/local/energyplus-9.5.0",
    model_file="EnergyPlus/5Zone/5ZoneAirCooled.idf",
    weather_file="EnergyPlus/Model-9-5-0/WeatherData/USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw",
    log_dir="eplog/ppo-year",
    seed=42,
    verbose=False,
    max_episode_steps=2880,
)

@wrap_experiment(log_dir="./data/local/garage/eplus_ppo_year", snapshot_mode="none")
def train(ctxt=None, seed=42):
    set_seed(seed)
    # Garage VPG/PPO uses CPU tensors internally; keep policy/value on CPU.
    env = GymEnv(EplusYearEnv(), max_episode_length=EplusYearEnv.max_episode_steps)
    trainer = Trainer(ctxt)
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(64, 64),
    )
    value_function = GaussianMLPValueFunction(
        env_spec=env.spec,
        hidden_sizes=(64, 64),
    )
    sampler = LocalSampler(
        agents=policy,
        envs=env,
        max_episode_length=env.spec.max_episode_length,
        worker_class=DefaultWorker,
    )
    algo = PPO(
        env_spec=env.spec,
        policy=policy,
        value_function=value_function,
        sampler=sampler,
        discount=0.99,
        gae_lambda=0.95,
        lr_clip_range=0.2,
        policy_ent_coeff=0.01,      # 熵系数
        entropy_method='regularized',       
    )

    trainer.setup(algo=algo, env=env)
    trainer.train(n_epochs=300, batch_size=5760)

if __name__ == "__main__":
    train()
