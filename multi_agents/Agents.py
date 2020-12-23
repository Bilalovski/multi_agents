import numpy


class Agents:

    def __init__(self, maxAge, energy, width, height, index):
        self.x = numpy.random.randint(width + 1)
        self.y = numpy.random.randint(height + 1)
        self.index = index
        self.maxAge = maxAge
        self.energy = energy
        self.age = 0
        self.width = width
        self.height = height
        self.will_die = False
        self.closest_enemy_loc = (0, 0)

    def printAll(self):
        print(self.x, self.y, self.maxAge, self.age, self.energy)

    def get_location(self):
        return [self.x, self.y]

    def is_dead(self):
        if self.energy <= 0 or self.age >= self.maxAge:
            # print("agent will die at age %d, %d", self.age, self.energy)
            return True
        else:
            return False


    def collision(self, agent):
        loc = self.get_location()
        if loc == agent.get_location():
            #print("prey dieded")
            return True
        else:
            return False
