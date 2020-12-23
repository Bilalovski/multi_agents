from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from ray.rllib.agents import with_common_config
from ray.rllib.agents.trainer_template import build_trainer

from dqn.dqn_policy import DQNPolicy

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = with_common_config({
    ########################################
    # Parameters Agent
    ########################################
    "lr": 0,
    "decay": 200,
    "gamma": 0.99,
    "target_update": 10,
    "hidden_nodes": 128,
    "train_batch_size": 128,

    "exp_mem": 2000,
    "dqn_model": {
        "custom_model": "?",
        "custom_model_config": {
        },  # extra options to pass to your model
    }
})
DQNTrainer = build_trainer(
    name="DQNAlgorithm",
    default_policy=DQNPolicy,
    default_config=DEFAULT_CONFIG)
