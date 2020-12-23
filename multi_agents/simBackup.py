import numpy
import pygame
import time
import random
import json
import matplotlib.pyplot as plt
import matplotlib
from Model import Model
from IPython import display


class Simulation:

    def __init__(self):
        print('init')
        self.model = Model()

        print('hello')
        # def readParameters():

        with open('/home/bilal/PycharmProjects/Simulation/config.json') as f:
            self.data = json.load(f)

        self.hunters_amount = int(self.data.get("amount_hunters"))
        self.prey_amount = int(self.data.get("amount_prey"))
        self.prey_birth_rate = float(self.data.get("prey birth-rate"))
        self.prey_max_age = int(self.data.get("prey max-age"))
        self.hunter_energy_to_reproduce = int(self.data.get("hunter energy-to-reproduce"))
        self.hunter_energy_per_prey_eaten = int(self.data.get("hunter energy-per-prey-eaten"))
        self.hunter_max_age = int(self.data.get("hunter max-age"))
        self.width = int(self.data.get("width"))
        self.height = int(self.data.get("height"))
        self.screen = pygame.display.set_mode([int(self.width), int(self.height)])
        self.is_ipython = 'inline' in matplotlib.get_backend()
        self.steps = 1
        self.to_plot = [[], []]
        self.prey_kids = 0
        self.hunter_kids = 0
        # procreation parameter, set to a value but be careful of energy needed for procreation and max age and max energy
        self.procreation_allowed = 60
        self.energy = self.data.get("hunter_energy")
        self.prey_env = None

        print("---------------PARAMETERS----------------")

        print(self.data)

        print("---------------PARAMETERS----------------")

    def run(self):
        # init pygame
        pygame.init()
        print("initialisation complete")

        self.add_hunter(self.hunters_amount)
        self.add_prey(self.prey_amount)

        print("hunters and prey made")

        # Run until the user asks to quit

        running = True

        while running:

            self.add_hunter(self.hunter_kids)
            self.add_prey(self.prey_kids)
            self.prey_kids = 0
            self.hunter_kids = 0
            # Did the user click the window close button?
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Fill the background with green
            self.screen.fill((255, 255, 255))

            self.visualise()

            self.step()

            time.sleep(0.033)

            # Flip the display
            pygame.display.flip()

            self.plot_environment()

        # Done! Time to quit.
        pygame.quit()

    def visualise(self):
        for i in self.model.return_hunter_list():
            loc = i.get_location()
            pygame.draw.circle(self.screen, (255, 0, 0), (loc[0], loc[1]), 1)
        for i in self.model.return_prey_list():
            loc = i.get_location()
            pygame.draw.circle(self.screen, (0, 255, 0), (loc[0], loc[1]), 1)

    def step(self):
        for hunter in self.model.return_hunter_list():
            self.hunter_kids += hunter.step(numpy.random.randint(5))
            if hunter.is_dead():
                self.model.kill_hunter(hunter)


        for prey in self.model.return_prey_list():
            prey.step(numpy.random.randint(4))
            if prey.is_dead():
                self.model.kill_prey(prey)
            if prey.can_procreate() and (self.steps % self.procreation_allowed == 0):
                self.prey_kids += int(self.prey_birth_rate * 100)

        for prey in self.model.return_prey_list():
            for hunter in self.model.return_hunter_list():
                if prey.collision(hunter):
                    hunter.energy += self.hunter_energy_per_prey_eaten
                    self.model.kill_prey(prey)

        self.steps += 1

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



    def add_hunter(self, amount):
        for i in range(amount):
            self.model.add_hunter(self.hunter_max_age, 100, self.hunter_energy_per_prey_eaten,
                                  self.hunter_energy_to_reproduce, self.width, self.height, self.prey_env)

    def add_prey(self, amount):
        for i in range(amount):
            self.model.add_prey(self.prey_max_age, 100, self.prey_birth_rate, self.width, self.height, self.prey_env)



def reset(self):
    self.model.reset_prey()
