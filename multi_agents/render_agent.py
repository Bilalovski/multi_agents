from time import sleep

import ray
import json
import gym
import numpy as np

from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune import register_env

from dqn import DQNTrainer, DQNModel
from prey_env import prey_env
from predator_env import predator_env

if __name__ == "__main__":

    # Settings
    folder = "/home/bilal/ray_results/DQNAlgorithm/DQNAlgorithm_pygame-v0_6c38c_00000_0_2020-11-15_22-35-03"
    env_name = "PreyEnv"
    checkpoint = 20
    num_episodes = 100

    def env_creator(env_config):
        return predator_env()

    register_env("PreyEnv", env_creator)

    # Def env
    #env = gym.make(env_name)
    env = predator_env()
    print(folder + "/params.json")

    ray.init()
    ModelCatalog.register_custom_model("DQNModel", DQNModel)

    # Load config
    with open(folder + "/params.json") as json_file:
        config = json.load(json_file)
    trainer = DQNTrainer(env=env_name, 
                         config=config)
    # Restore checkpoint
    config["decay"] = 1000000
    trainer.restore(folder + "/checkpoint_{}/checkpoint-{}".format(checkpoint, checkpoint))

    avg_reward = 0
    for episode in range(num_episodes):
        step = 0
        total_reward = 0
        done = False
        observation = env.reset()

        while not done:
            step += 1
            env.render()
            #print(observation)
            obs_batch = []
            for obs in observation.values():
                #print(obs)
                obs = np.array(obs)
                #print(obs)
                obs_batch.append(obs)
            #print(obs_batch)
            actions, _, _ = trainer.get_policy().compute_actions(obs_batch, [])
            action_dict = {}
            for i, action in enumerate(actions):
                index = "agent_" + str(i+1)
                action_dict[index] = action
            #print(action_dict)
            observation, rewards, done, info = env.step(action_dict)
            done = done.get("__all__")
            #print(done)
            reward = 0
            for r in rewards.values():
                reward = r
            total_reward += reward
            sleep(0.1)
        print("episode {} received reward {} after {} steps".format(episode, total_reward, step))
        avg_reward += total_reward
    print('avg reward after {} episodes is {}'.format(num_episodes, avg_reward/num_episodes))
    #env.close()
    del trainer