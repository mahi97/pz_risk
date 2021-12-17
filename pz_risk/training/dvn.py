### DVN ###
### Deep Value Network ###
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import math
import random
import numpy as np

from itertools import count
from collections import namedtuple

from training.utils import *

Transition = namedtuple('Transition', ('before_feat', 'before_adj', 'reward', 'feat', 'adj', 'done'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class GNN(nn.Module):
    def __init__(self, transform, activation):
        super(GNN, self).__init__()

        self.transform = transform
        self.activation = activation

    def forward(self, feat, adj):
        seq = self.transform(feat)
        ret = torch.matmul(adj, seq)
        return self.activation(ret)


class DVN(nn.Module):
    def __init__(self, feat_space, hidden_size):
        super(DVN, self).__init__()

        self.h1 = GNN(init_(nn.Linear(feat_space, hidden_size)), nn.ReLU())
        self.h2 = GNN(init_(nn.Linear(hidden_size, hidden_size)), nn.ReLU())
        self.h3 = GNN(init_(nn.Linear(hidden_size, hidden_size)), nn.ReLU())

        self.critic_linear = init_(nn.Linear(hidden_size, hidden_size))
        self.critic_linear2 = init_(nn.Linear(hidden_size, 1))
        self.train()

    def forward(self, feat, adj):
        hidden = self.h1(feat, adj)
        hidden = self.h2(hidden, adj)
        hidden = self.h3(hidden, adj)
        hidden = self.critic_linear(hidden)
        return torch.tanh(self.critic_linear2(hidden))


class DVNAgent(nn.Module):
    def __init__(self, num_nodes, num_agents, feat_size, hidden_size, device='cuda:0'):
        super(DVNAgent, self).__init__()
        self.device = device
        self.policy_network = DVN(feat_size, hidden_size).to(device)
        self.target_network = DVN(feat_size, hidden_size).to(device)

        self.memory = ReplayMemory(2 ** 10)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.RMSprop(self.policy_network.parameters())

        self.batch_size = 100
        self.gamma = 0.999
        self.target_update = 20
        self.num_train = 0
        self.n_nodes = num_nodes
        self.n_agents = num_agents
        self.feat_size = feat_size

    def forward(self, feat, adj):
        with torch.no_grad():
            return self.policy_network(feat, adj)

    def train_(self):
        self.num_train += 1
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask_feat = torch.tensor(tuple(map(lambda s: s is not None, batch.feat)))
        # non_final_mask_adj = torch.tensor(tuple(map(lambda s: s is not None, batch.adj)), torch.bool, self.device)
        non_final_next_feat = torch.cat([s for s in batch.feat if s is not None]).view(-1, self.n_nodes + self.n_agents, self.feat_size)
        non_final_next_adj = torch.cat([s for s in batch.adj if s is not None]).view(-1, self.n_nodes + self.n_agents, self.n_nodes + self.n_agents)

        feat_batch = torch.cat(batch.before_feat).view(-1, self.n_nodes + self.n_agents, self.feat_size)
        adj_batch = torch.cat(batch.before_adj).view(-1, self.n_nodes + self.n_agents, self.n_nodes + self.n_agents)
        reward_batch = torch.cat(batch.reward)

        q = self.policy_network(feat_batch, adj_batch).squeeze()[:, self.n_nodes:]

        y = torch.zeros([self.batch_size, self.n_agents], device=self.device)
        y[non_final_mask_feat] = self.target_network(non_final_next_feat, non_final_next_adj)[:, self.n_nodes:].squeeze().detach() * self.gamma
        y += reward_batch
        # print(y, reward_batch)
        # Compute Huber loss
        loss = F.smooth_l1_loss(y, q)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if self.num_train % self.target_update == 0:
            self.update_target()

        return loss.item()

    def train_start(self):
        return len(self.memory) > self.batch_size  # Boolean

    def update_target(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

    def save_memory(self, transition):
        # transition[1] = torch.tensor([[transition[1]]], device=self.device, dtype=torch.long)  # Action
        # transition[2] = torch.tensor([transition[2]], device=self.device)  # Reward
        self.memory.push(*transition)
