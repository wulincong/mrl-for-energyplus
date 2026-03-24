from garage import Trainer, wrap_experiment
from garage.trainer import TFTrainer
from garage.experiment.deterministic import set_seed
from garage.envs import GymEnv

@wrap_experiment
def my_first_experiment(ctxt=None, seed=1):
    set_seed(seed)
    trainer = Trainer(ctxt)
    
    with TFTrainer as trainer:
        env = GymEnv()

    