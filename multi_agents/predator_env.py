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


class predator_env(gym.Env, MultiAgentEnv):

    def __init__(self):
        print("initialising evironment")
        self.rewards = {}
        self.count = 0
        self.simulator = Simulation()
        self.action_space = Discrete(5)
        self.done = {"__all__": False}
        self.MAX_STEPS = 1000
        self.obs_dictionary = {}
        self.observation_space = spaces.Box(np.array([0, 0, 0]),
                                            np.array([self.simulator.hunter_max_age * 30, self.simulator.width * 30,
                                                      self.simulator.height * 30]))

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
        hunter_loc = np.array([x, y])

        curr_min_val = math.sqrt(pow(self.simulator.height, 2) + pow(self.simulator.width, 2))
        curr_min_x = None
        curr_min_y = None
        for prey in self.simulator.model.prey_list:
            prey_loc = np.array([prey.x, prey.y])
            dist = np.linalg.norm(prey_loc - hunter_loc)
            if dist < curr_min_val:
                curr_min_x = prey_loc[0]
                curr_min_y = prey_loc[1]
                curr_min_val = dist

        return curr_min_x, curr_min_y

    def reset(self):
        # print("resetting environment")
        # reward moet nog gereset worden
        # self.state, reward, self.done, self.info
        obs_dict = self.simulator.reset_predator()
        self.done.clear()
        self.obs_dictionary.clear()
        self.rewards.clear()
        self.done.update({"__all__": False})
        for hunter in self.simulator.model.hunter_list:
            hunter.x, hunter.y = self.who_is_closest(hunter.x, hunter.y)
        # print("environment reset")
        return obs_dict

    def step(self, action):
        # print(self.done)
         
        for hunter in self.simulator.model.hunter_list:
            if hunter.will_die:
                self.simulator.model.kill_hunter(hunter)
        for hunter in self.simulator.model.hunter_list:
            # print(prey.closest_enemy_loc)
            
            if hunter.is_dead():
                hunter.will_die = True
            hunter.closest_enemy_loc = self.who_is_closest(hunter.x, hunter.y)
        """for hunter in self.simulator.model.hunter_list:
            print("hunter location")
            print(hunter.x, hunter.y)
        """
        # print("stepping")
        self.observation_space, self.rewards, dones = self.simulator.hunter_step(action)
        # print(self.observation_space)
        self.done.update(dones)
        self.simulator.prey_step_random()
        # print("stepped")
        """print("hunter list")
        print(len(self.simulator.model.hunter_list))
        print("prey list")
        print(len(self.simulator.model.prey_list))"""
        if len(self.observation_space) == 0:
            print("DEAD")
            print("ate %d amount of prey", self.simulator.total_prey_eaten)
            # self.simulator.plot_prey_eaten()
            self.done.update({"__all__": True})
        elif len(self.simulator.model.prey_list) <= 0:
            self.done.update({"__all__": True})
            # self.simulator.plot_prey_eaten()
            print("Win")
            print("ate %d amount of prey", self.simulator.total_prey_eaten)

        elif self.simulator.model.get_prey_amount() + self.simulator.model.get_hunter_amount() > 40000:
            self.done.update({"__all__": True})
            # self.simulator.plot_prey_eaten()
            print("tie")

        # self.simulator.plot_hunter_prey_amount()
        # print(self.rewards)

        return self.observation_space, self.rewards, self.done, {}

    """def render(self, mode='human', close=False):
        self.simulator.visualise()
"""