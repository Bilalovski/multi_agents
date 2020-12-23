import random
from pathlib import Path
import numpy
import json
# import pygame
from Model import Model
# from IPython import display
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


class Simulation:

    def __init__(self):
        path = Path(__file__).parent / "config.json"
        print(path)
        with open(path) as f:
            self.data = json.load(f)
        print('init')
        self.model = Model()

        print('hello')
        # def readParameters():

        self.hunters_amount = int(self.data.get("amount_hunters"))
        self.prey_amount = int(self.data.get("amount_prey"))
        self.prey_birth_rate = float(self.data.get("prey birth-rate"))
        self.prey_max_age = int(self.data.get("prey max-age"))
        self.hunter_energy_to_reproduce = int(self.data.get("hunter energy-to-reproduce"))
        self.hunter_energy_per_prey_eaten = int(self.data.get("hunter energy-per-prey-eaten"))
        self.hunter_max_age = int(self.data.get("hunter max-age"))
        self.width = int(self.data.get("width"))
        self.height = int(self.data.get("height"))
        #self.screen = pygame.display.set_mode([int(self.width), int(self.height)])
        # self.is_ipython = 'inline' in matplotlib.get_backend()
        self.steps = 1
        self.to_plot = [[], [], []]
        self.prey_kids = 0
        self.hunter_kids = 0
        # procreation parameter, set to a value but be careful of energy needed for procreation and max age and max energy
        self.procreation_allowed = 60
        self.energy = int(self.data.get("energy"))
        self.prev = 0
        self.total_prey_eaten = 0
        """
        print("---------------PARAMETERS----------------")

        print(self.data)

        print("---------------PARAMETERS----------------")
        """
        #pygame.init()

    """def visualise(self):
        self.screen.fill((255, 255, 255))
        for i in self.model.hunter_list:
            loc = i.get_location()
            pygame.draw.circle(self.screen, (255, 0, 0), (loc[0], loc[1]), 1)
        for i in self.model.return_prey_list():
            loc = i.get_location()
            pygame.draw.circle(self.screen, (0, 255, 0), (loc[0], loc[1]), 1)
        pygame.display.flip()"""

    def prey_step(self, actions):
        self.make_kids(self.prey_kids, self.hunter_kids)
        # print("stepping in simulator")
        action_data = list(actions.items())
        actions_array = numpy.array(action_data)
        # print("will print actions:")
        # print(actions)
        index_array = []
        obs_dict = {}
        reward_dict = {}
        dones = {}
        for data in actions_array:
            index_array.append(data[0])
        # door alle prooien loopen en zien of ze de juiste index hebben, zo ja, voer actie uit
        for prey in self.model.prey_list:
            # print("the key of the action dictionary")
            chance = random.random()
            if chance < self.prey_birth_rate:
                self.prey_kids += 1

            agent = "prey_" + str(prey.index)
            # print(i)
            # print("prey list:")
            for i in index_array:
                if "prey_" + str(prey.index) == i or not index_array.__contains__("prey_" + str(prey.index)):
                    # print("agent found in the list")
                    obs = prey.step(actions.get(i))
                    # print("step function in prey class done")
                    obs_dict.update({agent: obs})
                    # print("dictionary updated")
                    reward_dict.update({agent: self.calc_prey_reward(prey)})
                    for hunter in self.model.hunter_list:
                        if prey.collision(hunter) or prey.is_dead():
                            # dones.update({agent: True})
                            # self.model.kill_prey(prey)
                            prey.will_die = True
                        else:
                            dones.update({agent: False})

                    break

            # print("simulator stepping done")

        return obs_dict, reward_dict, dones

    def hunter_step(self, actions):
        self.make_kids(self.prey_kids, self.hunter_kids)

        # print("stepping in simulator")
        action_data = list(actions.items())
        actions_array = numpy.array(action_data)
        # print("will print actions:")
        # print(actions)
        index_array = []
        obs_dict = {}
        reward_dict = {}
        dones = {}
        # print(len(self.model.hunter_list))
        # print(actions)
        for data in actions_array:
            index_array.append(data[0])
        # door alle hunters loopen en zien of ze de juiste index hebben, zo ja, voer actie uit
        for hunter in self.model.hunter_list:
            # print("the key of the action dictionary")
            agent = "hunter_" + str(hunter.index)
            # print(i)
            # print("prey list:")
            for i in index_array:
                if "hunter_" + str(hunter.index) == i or not index_array.__contains__("hunter_" + str(hunter.index)):
                    # print("agent found in the list")
                    obs = hunter.step(actions.get(i))
                    if actions.get(i) == 4:
                        if hunter.procreate:
                            hunter.children_made += 1
                            self.hunter_kids += 1
                            hunter.energy -= self.hunter_energy_to_reproduce
                            hunter.procreate = False
                    # print("step function in prey class done")
                    obs_dict.update({agent: obs})
                    # print("dictionary updated")
                    reward_dict.update({agent: self.calc_hunter_reward(hunter)})
                    for prey in self.model.prey_list:
                        if hunter.collision(prey):
                            prey.eaten = True
                            hunter.prey_eaten += 1
                            self.total_prey_eaten += 1
                            hunter.energy += self.hunter_energy_per_prey_eaten

                    break

            # print("simulator stepping done")

        return obs_dict, reward_dict, dones

    def hunter_step_random(self):
        for hunter in self.model.return_hunter_list():
            hunter.step(numpy.random.randint(5))
            if hunter.procreate:
                hunter.procreate = False
                self.hunter_kids += 1
                # print("hunter added")
            if hunter.is_dead():
                self.model.kill_hunter(hunter)

    def prey_step_random(self):
        for prey in self.model.return_prey_list():
            prey.step(numpy.random.randint(4))
            if prey.is_dead() or prey.eaten:
                self.model.kill_prey(prey)
            chance = random.random()
            if chance <= self.prey_birth_rate:
                self.prey_kids += 1
        """
    def plot_environment(self):
        plt.figure(2)
        plt.clf()
        plt.title('prey_hunter_sim')
        plt.xlabel('time step')
        plt.ylabel('amount')
        self.to_plot[0].append(self.model.get_prey_amount())
        self.to_plot[1].append(self.model.get_hunter_amount())
        plt.plot(self.to_plot[0], label='prey')
        plt.plot(self.to_plot[1], label='hunter')
        plt.legend()
        plt.pause(0.001)  # pause a bit so that plots are updated
        if self.is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())
        """

    def reset_prey(self):
        self.model.prey_index = 1
        self.model.hunter_index = 1
        self.prev = 0
        del self.model.hunter_list[:]
        for i in range(self.hunters_amount):
            self.model.add_hunter(self.hunter_max_age, self.energy, self.hunter_energy_per_prey_eaten,
                                  self.hunter_energy_to_reproduce, self.width, self.height)
        del self.model.prey_list[:]
        for i in range(self.prey_amount):
            self.model.add_prey(self.prey_max_age, self.energy, self.prey_birth_rate, self.width, self.height)
        obs_dict = {}
        for prey in self.model.prey_list:
            obs = [prey.age, prey.x, prey.y]
            agent = "prey_" + str(prey.index)
            obs_dict.update({agent: obs})
        return obs_dict

    def reset_predator(self):
        self.model.prey_index = 1
        self.model.hunter_index = 1
        del self.model.hunter_list[:]
        for i in range(self.hunters_amount):
            self.model.add_hunter(self.hunter_max_age, self.energy, self.hunter_energy_per_prey_eaten,
                                  self.hunter_energy_to_reproduce, self.width, self.height)
        del self.model.prey_list[:]
        for i in range(self.prey_amount):
            self.model.add_prey(self.prey_max_age, self.energy, self.prey_birth_rate, self.width, self.height)
        obs_dict = {}
        for hunter in self.model.hunter_list:
            temp = hunter.closest_enemy_loc
            obs = [hunter.age, temp[0], temp[1]]
            agent = "hunter_" + str(hunter.index)
            obs_dict.update({agent: obs})
        self.total_prey_eaten = 0
        return obs_dict

    def make_kids(self, prey_am, hunter_am):
        for i in range(prey_am):
            self.model.add_prey(self.prey_max_age, self.energy, self.prey_birth_rate, self.width, self.height)
        for i in range(hunter_am):
            self.model.add_hunter(self.hunter_max_age, self.energy, self.hunter_energy_per_prey_eaten,
                                  self.hunter_energy_to_reproduce, self.width, self.height)
        self.hunter_kids = 0
        self.prey_kids = 0

    def calc_hunter_reward(self, hunter):
        reward2 = hunter.energy / 50 + 2 * hunter.procreate
        hunter.procreate = False
        hunter.prey_eaten = 0
        hunter.children_made = 0
        return reward2

    def plot_hunter_prey_amount(self):
        plt.figure(3)
        plt.clf()
        plt.title('prey_hunter_sim')
        plt.xlabel('time step')
        plt.ylabel('amount')
        self.to_plot[0].append(self.model.get_prey_amount())
        self.to_plot[1].append(self.model.get_hunter_amount())
        plt.plot(self.to_plot[0], label='prey')
        plt.plot(self.to_plot[1], label='hunter')
        plt.legend()
        plt.pause(0.001)

    def plot_prey_eaten(self):
        plt.figure(1)
        plt.clf()
        plt.title('amount of prey eaten')
        plt.xlabel('time step')
        plt.ylabel('amount')
        self.to_plot[2].append(self.total_prey_eaten)
        plt.plot(self.to_plot[2], label='prey eaten')
        plt.legend()
        plt.pause(0.001)

    def calc_prey_reward(self, prey):
        loc = [prey.x, prey.y]
        prey_loc = np.array(loc)
        hunter_loc = np.array(prey.closest_enemy_loc)

        return np.linalg.norm(prey_loc - hunter_loc)
