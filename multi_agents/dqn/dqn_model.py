from torch import nn, cat
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from gym.spaces import Discrete, Box
import torch.nn.functional as F


class DQNModel(nn.Module, TorchModelV2):

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, num_hidden_nodes):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)
        self.obs_space = obs_space
        self.action_space = action_space
        self.model_config = model_config
        self.name = name
        self.hidden_nodes = num_hidden_nodes

        if isinstance(self.obs_space, Box):
            self.obs_shape = obs_space.shape[0]
        else:
            self.obs_shape = self.obs_space

        self.layers = nn.Sequential()
        self.layers.add_module("linear_1", nn.Linear(self.obs_shape, self.hidden_nodes))
        self.layers.add_module("relu_1", nn.LeakyReLU())
        self.layers.add_module("linear_2", nn.Linear(self.hidden_nodes, self.hidden_nodes))
        self.layers.add_module("Relu_2", nn.LeakyReLU())
        self.layers.add_module("linear_3", nn.Linear(self.hidden_nodes, self.hidden_nodes))
        self.layers.add_module("Relu_3", nn.LeakyReLU())
        self.layers.add_module("linear_4", nn.Linear(self.hidden_nodes, self.hidden_nodes))
        self.layers.add_module("Relu_4", nn.LeakyReLU())
        self.layers.add_module("linear_5", nn.Linear(self.hidden_nodes, self.hidden_nodes))
        self.layers.add_module("Relu_5", nn.LeakyReLU())
        self.layers.add_module("linear_6", nn.Linear(self.hidden_nodes, num_outputs))



    @override(TorchModelV2)
    def forward(self, obs):
        return self.layers(obs)
