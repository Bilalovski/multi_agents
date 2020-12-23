from Agents import Agents


class Hunter(Agents):
    def __init__(self, maxAge, energy, epp, etr, width, height, index):
        Agents.__init__(self, maxAge, energy, width, height, index)
        self.epp = epp
        self.etr = etr
        self.prey_eaten = 0
        self.children_made = 0
        self.procreate = False

    def step(self, action):
        v = []
        if action == 0:  # left
            v = [-1, 0]
        if action == 1:  # right
            v = [1, 0]
        if action == 2:  # up
            v = [0, -1]
        if action == 3:  # down
            v = [0, 1]
        if action == 4:  # reproduce
            if self.can_procreate():
                self.procreate = True
        if not action == 4:
            if 0 <= self.x + v[0] <= self.width:
                self.x = self.x + v[0]
            if 0 <= self.y + v[1] <= self.height:
                self.y = self.y + v[1]

        self.energy -= 1
        self.age += 1


        return [self.age, self.closest_enemy_loc[0], self.closest_enemy_loc[1]]

    def can_procreate(self):
        if self.energy >= self.etr:
            return True
        else:
            return False
