import math

import gym as gym
from IPython import display
from gym.spaces import Box, Discrete
from ray.rllib.env import MultiAgentEnv
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from Simulation import Simulation
import random


class prey_env(gym.Env, MultiAgentEnv):

    def __init__(self):
        print("initialising evironment")
        self.rewards = {}
        self.count = 0
        self.simulator = Simulation()
        self.action_space = Discrete(4)
        self.done = {"__all__": False}
        self.MAX_STEPS = 1000
        self.obs_dictionary = {}
        self.observation_space = spaces.Box(np.array([0, 0, 0]),
                                            np.array([self.simulator.prey_max_age * 30, self.simulator.width * 30,
                                                      self.simulator.height * 30]))
        self.done.update({"no_done_at_end": True})

        print("environment initialised")

    def plot(self):
        print("plotting now")
        plt.figure(10)
        plt.clf()
        plt.title('prey_hunter_sim')
        plt.xlabel('time step')
        plt.ylabel('amount')
        self.to_plot[0].append(self.simulator.model.get_prey_amount())
        self.to_plot[1].append(self.simulator.model.get_hunter_amount())
        plt.plot(self.simulator.model.get_prey_amount(), label='prey')
        plt.plot(self.simulator.model.get_hunter_amount(), label='hunter')
        plt.legend()
        plt.pause(0.001)  # pause a bit so that plots are updated
        if self.is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())

    def who_is_closest(self, x, y):
        prey_loc = np.array([x, y])

        curr_min_val = math.sqrt(pow(self.simulator.height, 2) + pow(self.simulator.width, 2))
        curr_min_x = None
        curr_min_y = None
        for hunter in self.simulator.model.hunter_list:
            hunter_loc = np.array([hunter.x, hunter.y])
            dist = np.linalg.norm(prey_loc - hunter_loc)
            if dist < curr_min_val:
                curr_min_x = hunter_loc[0]
                curr_min_y = hunter_loc[1]
                curr_min_val = dist

        return curr_min_x, curr_min_y

    def reset(self):
        # print("resetting environment")
        # reward moet nog gereset worden
        # self.state, reward, self.done, self.info
        obs_dict = self.simulator.reset_prey()
        self.done.clear()
        self.obs_dictionary.clear()
        self.rewards.clear()
        self.done.update({"no_done_at_end": True})
        self.done.update({"__all__": False})
        # print("environment reset")
        return obs_dict

    def step(self, action):
        # print(self.done)
        print("do something")
        for prey in self.simulator.model.prey_list:
            # print(prey.closest_enemy_loc)
            prey.closest_enemy_loc = self.who_is_closest(prey.x, prey.y)
            if prey.will_die:
                self.done.update({"agent_" + str(prey.index): True})
                self.simulator.model.kill_prey(prey)
        """for hunter in self.simulator.model.hunter_list:
            print("hunter location")
            print(hunter.x, hunter.y)
        """
        chance = random.random()
        # print("stepping")
        self.observation_space, self.rewards, dones = self.simulator.prey_step(action)
        self.done.update(dones)
        self.simulator.hunter_step_random()
        print("ripperoni")
        # print("stepped")
        """print("hunter list")
        print(len(self.simulator.model.hunter_list))
        print("prey list")
        print(len(self.simulator.model.prey_list))"""

        if len(self.simulator.model.prey_list) == 0:
            print("DEDED")
            self.done.update({"__all__": True})
        elif len(self.simulator.model.hunter_list) == 0:
            self.done.update({"__all__": True})
            print("Win lmao")

        if chance > self.simulator.prey_birth_rate:
            self.simulator.prey_kids += 1


        return self.observation_space, self.rewards, self.done, {}

    def render(self, mode='human', close=False):
        self.simulator.visualise()

