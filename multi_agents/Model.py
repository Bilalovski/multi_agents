from Hunter import Hunter
from Prey import Prey


class Model:
    def __init__(self):
        self.hunter_list = []
        self.prey_list = []
        self.agent_list = []
        self.prey_index = 1
        self.hunter_index = 1

    def add_hunter(self, maxAge, energy, epp, etr, width, height):
        hunter = Hunter(maxAge, energy, epp, etr, width, height, self.hunter_index)
        self.hunter_list.append(hunter)
        self.agent_list.append(hunter)
        self.hunter_index += 1
        return hunter

    def add_prey(self, maxAge, energy, br, width, height):
        prey = Prey(maxAge, energy, br, width, height, self.prey_index)
        self.prey_list.append(prey)
        self.agent_list.append(prey)
        self.prey_index += 1
        return prey

    def get_hunter_amount(self):
        return len(self.hunter_list)

    def get_prey_amount(self):
        return len(self.prey_list)

    def kill_prey(self, prey):
        if prey in self.prey_list:
            self.prey_list.remove(prey)

    def kill_hunter(self, hunter):
        if hunter in self.hunter_list:
            self.hunter_list.remove(hunter)

    def return_hunter_list(self):
        return self.hunter_list

    def return_prey_list(self):
        return self.prey_list

    def return_agent_list(self):
        return self.agent_list

    def kill_agent(self, agent):
        self.agent_list.remove(agent)
        if agent in self.prey_list:
            self.prey_list.remove(agent)
        if agent in self.hunter_list:
            self.hunter_list.remove(agent)


