#import pygame

from Agents import Agents

class Prey(Agents):
    def __init__(self, maxAge, energy, br, width, height, index):
        Agents.__init__(self, maxAge, energy, width, height, index)

        self.br = br
        self.eaten = False

    def step(self, action):
        v = []
        # print(action)
        if action == 0:  # left
            v = [-1, 0]
        if action == 1:
            v = [1, 0]
        if action == 2:
            v = [0, -1]
        if action == 3:
            v = [0, 1]

        #print(action)

        if 0 <= self.x + v[0] <= self.width:
            self.x = self.x + v[0]
        if 0 <= self.y + v[1] <= self.height:
            self.y = self.y + v[1]
        # print(self.x, self.y)
        self.energy -= 1
        self.age += 1
        """print("closest enemy loc according to prey: ")
        print(self.closest_enemy_loc)"""
        return [self.age, self.closest_enemy_loc[0], self.closest_enemy_loc[1]]

