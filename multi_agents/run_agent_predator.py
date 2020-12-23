import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from dqn import DQNTrainer, DQNModel
from prey_env import prey_env
from predator_env import predator_env


def env_creator(env_config):
    return predator_env()


if __name__ == "__main__":
    ray.init()
    ModelCatalog.register_custom_model("DQNModel", DQNModel)
    register_env("predator_env", env_creator)
    print("will run tune")

    tune.run(
        DQNTrainer,
        checkpoint_freq=5,
        checkpoint_at_end=True,
        stop={"episodes_total": 150},
        # stop={"episode_reward_mean": 300},
        config={
            "num_gpus": 0,
            "num_workers": 1,
            "framework": "torch",
            "rollout_fragment_length": 10,
            "env": "predator_env",
            "no_done_at_end": True,
            "train_batch_size": 128,
            ########################################
            # Parameters Agent
            ########################################
            "lr": 0.000001,
            # "lr": tune.grid_search([0.000001, 0.00001, 0.0001, 0.0005]),
            # ----------------------------------------------------------
            # "decay": tune.grid_search([0.01, 0.001, 0.0001, 0.00001, 0.00005]),
            "decay": 0.00001,
            # "batch_size": 1,
            # "batch_size": tune.grid_search([10, 50, 100]),
            "gamma": 0.99,
            # "gamma": tune.grid_search([0.7, 0.9, 0.99, 0.999]),
            # "exp_mem": 20000,
            # "exp_mem": tune.grid_search([10000, 20000]),

            # "target_update": tune.grid_search([40, 50, 60, 70]),
            "target_update": 1,

            "hidden_nodes": 600,
            # ----------------------------------------------------------

            "dqn_model": {
                "custom_model": "DQNModel",
                "custom_model_config": {
                },  # extra options to pass to your model
            }
        }
    )
    print("tune ran")
