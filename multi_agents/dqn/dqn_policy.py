import math

import torch
import random
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
from IPython import display
import matplotlib.pyplot as plt
import matplotlib

from ray.rllib.policy import Policy
from ray.rllib.models import ModelCatalog
from torch import optim, nn


class DQNPolicy(Policy):
    """Hand-coded policy that returns random actions."""

    def __init__(self, observation_space, action_space, config):
        print(observation_space, action_space.n)
        Policy.__init__(self, observation_space, action_space, config)
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config
        # ----------------------------------------------------------

        self.lr = self.config["lr"]  # Extra options need to be added in dqn.py
        self.iteration = 0
        self.to_plot = []
        self.hidden_nodes = config["hidden_nodes"]

        self.current_step = 0
        self.start = 1
        self.end = 0.005
        self.gamma = config["gamma"]
        # self.criterion = nn.MSELoss()

        self.decay = self.config["decay"]
        self.is_ipython = 'inline' in matplotlib.get_backend()
        if self.is_ipython:
            from IPython import display

        # ----------------------------------------------------------

        # GPU settings
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.device = torch.device("cuda")
            print("using cuda")
        else:
            self.device = torch.device("cpu")
            print("using cpu")

        self.policy_net = ModelCatalog.get_model_v2(
            obs_space=self.observation_space,
            action_space=self.action_space,
            num_outputs=self.action_space.n,
            name="DQNModel",
            model_config=self.config["dqn_model"],
            framework="torch",
            num_hidden_nodes=self.hidden_nodes,
        ).to(self.device, non_blocking=True)
        # -------------------------------------------------------
        self.target_net = ModelCatalog.get_model_v2(
            obs_space=self.observation_space,
            action_space=self.action_space,
            num_outputs=self.action_space.n,
            name="DQNModel",
            model_config=self.config["dqn_model"],
            framework="torch",
            num_hidden_nodes=self.hidden_nodes,
        ).to(self.device, non_blocking=True)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.target_update = config["target_update"]

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        # ---------------------------------------------------------

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        explore=None,
                        timestep=None,
                        **kwargs):
        # Worker function
        # action_space (links of rechts = 0 of 1)
        obs_batch_t = torch.tensor(obs_batch).type(torch.FloatTensor)
        obs_batch_a = np.array(obs_batch_t)
        # print("--------------obs batch------------")
        # print(obs_batch_a)
        # print("-----------------------------------")

        # exploration vs exploitation
        threshold = self.end + (self.start - self.end) * math.exp(-1. * self.current_step * self.decay)
        # hoe hoger decay, hoe lager threshold, hoe groter de kans op greedy
        action = []

        chance = random.random()
        """
        if chance > threshold:
            value = self.policy_net(obs_batch_t)
            action = torch.argmax(value).item()
            # print("not random")
        else:
            action = self.action_space.sample()
            # print("random")
        """
        if chance > threshold:
            print("exploit this shizzle")

            for i in range(np.shape(obs_batch_a)[0]):
                # print("exploit")
                value = self.policy_net(obs_batch_t)
                # print(value)
                # print(torch.argmax(value[i]).item())
                action.append(torch.argmax(value[i]).item())

        else:
            for i in range(np.shape(obs_batch_a)[0]):
                action.append(self.action_space.sample())
                # print("random")        # print(action)
        self.current_step += 1
        # print("action computed")

        return action, [], {}

    def learn_on_batch(self, samples):
        print("learning on batch")
        # Trainer function
        self.iteration += 1
        obs_batch_t = torch.tensor(np.array(samples["obs"])).type(torch.FloatTensor)
        rewards_batch_t = torch.tensor(np.array(samples["rewards"])).type(torch.FloatTensor)
        new_obs_batch_t = torch.tensor(np.array(samples["new_obs"])).type(torch.FloatTensor)
        done = torch.tensor(np.array(samples["dones"])).type(torch.FloatTensor)
        actions = torch.tensor(np.array(samples["actions"]))
        # print(np.asarray(obs_batch_t).shape)

        """
        for state, action, reward, next_state, done in zip(obs_batch_t, actions, rewards_batch_t, new_obs_batch_t,
                                                           done):
            self.exp_buffer.push(state, action, next_state, reward, done)
        """
        # transitions = self.exp_buffer.sample(self.batch_size)
        state_batch = obs_batch_t
        action_batch = actions.unsqueeze(1)
        reward_batch = rewards_batch_t
        next_state_batch = new_obs_batch_t
        dones = done

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_state_action_values = (next_state_values.unsqueeze(1) * self.gamma) + reward_batch.unsqueeze(1)

        for i in range(len(reward_batch)):
            if dones[i] == 1:
                expected_state_action_values[i] = reward_batch[i]

        loss = F.smooth_l1_loss(state_action_values.unsqueeze(1), expected_state_action_values.unsqueeze(1))
        # loss = self.criterion(expected_state_action_values, state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if self.iteration % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.to_plot.append(loss.detach())

        self.plot_durations()
        # print("learned")

        return {"learner_stats": {"loss": loss}}

    def get_weights(self):
        # Trainer function
        weights = {}
        weights["dqn_model"] = self.policy_net.cpu().state_dict()
        self.policy_net.to(self.device, non_blocking=False)
        return weights

    def set_weights(self, weights):
        # Worker function
        if "dqn_model" in weights:
            self.policy_net.load_state_dict(weights["dqn_model"], strict=True)
            self.policy_net.to(self.device, non_blocking=False)

    def plot_durations(self):
        plt.figure(2)
        plt.clf()
        plt.title('Training...')
        plt.xlabel('iterations ')
        plt.ylabel('loss')
        plt.plot(self.to_plot)

        plt.pause(0.001)  # pause a bit so that plots are updated
        if self.is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())
