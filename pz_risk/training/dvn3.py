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

from copy import deepcopy
from utils import *

Transition = namedtuple('Transition', ('before_feat', 'before_adj', 'reward', 'feat', 'adj', 'type', 'done'))


class ReplayMemory(object):

    def __init__(self, capacity, device, n_nodes, n_agents, feat_size):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.device = device
        self.n_nodes = n_nodes
        self.n_agents = n_agents
        self.feat_size = feat_size

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        samples = random.sample(self.memory, min(batch_size, len(self.memory)))
        batch = Transition(*zip(*samples))
        before_feat = torch.tensor(batch.before_feat, dtype=torch.float32, device=self.device).reshape(-1,
                                                                                                       self.n_nodes + self.n_agents,
                                                                                                       self.feat_size)
        before_adj = torch.tensor(np.array(batch.before_adj), dtype=torch.float32, device=self.device).reshape(-1,
                                                                                                               self.n_nodes + self.n_agents,
                                                                                                               self.n_nodes + self.n_agents)
        feat = torch.tensor(np.array(batch.feat), dtype=torch.float32, device=self.device).reshape(-1,
                                                                                                   self.n_nodes + self.n_agents,
                                                                                                   self.feat_size)
        adj = torch.tensor(np.array(batch.adj), dtype=torch.float32, device=self.device).reshape(-1,
                                                                                                 self.n_nodes + self.n_agents,
                                                                                                 self.n_nodes + self.n_agents)
        types = torch.tensor(np.array(batch.type), dtype=torch.float32, device=self.device).reshape(-1,
                                                                                                    self.n_nodes + self.n_agents)
        reward = torch.tensor(np.array(batch.reward), dtype=torch.float32, device=self.device).reshape(-1,
                                                                                                       self.n_agents)
        done = torch.tensor(np.array(batch.done), dtype=torch.bool, device=self.device).reshape(-1, self.n_agents)
        return Transition(before_feat, before_adj, reward, feat, adj, types, done)

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
    def __init__(self, feat_space, hidden_size, num_types):
        super(DVN, self).__init__()

        self.emb1 = GNN(init_(nn.Linear(feat_space, hidden_size)), nn.ReLU())
        # self.emb2 = GNN(init_(nn.Linear(feat_space, hidden_size)), nn.ReLU())
        # self.emb3 = GNN(init_(nn.Linear(feat_space, hidden_size)), nn.ReLU())
        # self.emb4 = GNN(init_(nn.Linear(feat_space, hidden_size)), nn.ReLU())
        # self.h1 = GNN(init_(nn.Linear(feat_space, hidden_size)), nn.ReLU())

        self.h2 = GNN(init_(nn.Linear(hidden_size, hidden_size)), nn.ReLU())
        self.h3 = GNN(init_(nn.Linear(hidden_size, hidden_size)), nn.ReLU())
        self.types = num_types
        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        self.train()

    def forward(self, feat, adj, types):
        n = self.types
        hidden = self.emb1(feat, adj)
        # h1 = self.emb1(feat[:, :n], adj[:, :n, :n])
        # h2 = self.emb2(feat[:, :n], adj[:, n:, :n])
        # h3 = self.emb3(feat[:, n:], adj[:, n:, n:])
        # h4 = self.emb4(feat[:, n:], adj[:, :n, n:])
        # hidden = torch.cat([(h1 + h4), (h2 + h3)], dim=1)
        hidden = self.h2(hidden, adj)
        hidden = self.h3(hidden, adj)
        return self.critic_linear(hidden)


class DVNAgent(nn.Module):
    def __init__(self, num_nodes, num_agents, num_types, feat_size, hidden_size, device='cuda:0'):
        super(DVNAgent, self).__init__()
        self.device = device
        self.policy_network = DVN(feat_size, hidden_size, num_types).to(device)
        self.target_network = DVN(feat_size, hidden_size, num_types).to(device)

        self.memory = ReplayMemory(2 ** 14, device, num_nodes, num_agents, feat_size)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.RMSprop(self.policy_network.parameters(), lr=1e-5)

        self.batch_size = 128
        self.gamma = 0.999
        self.target_update = 20
        self.num_train = 0
        self.n_nodes = num_nodes
        self.n_agents = num_agents
        self.feat_size = feat_size

    def get_value(self, board, agent_id):
        sim_feat, sim_adj, sim_type = get_feat_adj_type_from_board(board, self.n_agents)
        sim_feat = torch.tensor(sim_feat, dtype=torch.float32, device=self.device).reshape(-1,
                                                                                           self.n_nodes + self.n_agents,
                                                                                           self.feat_size)
        sim_adj = torch.tensor(sim_adj, dtype=torch.float32, device=self.device).reshape(-1,
                                                                                         self.n_nodes + self.n_agents,
                                                                                         self.n_nodes + self.n_agents)
        sim_type = torch.tensor(sim_type, dtype=torch.float32, device=self.device).reshape(-1,
                                                                                           self.n_nodes + self.n_agents)
        if len(board.player_nodes(agent_id)) == self.n_nodes:
            return [[10000]]
        else:
            return self.forward(sim_feat, sim_adj, sim_type).detach().cpu().numpy()[:, self.n_nodes + agent_id]

    def predict(self, board, agent_id):
        action_scores = []
        deterministic, valid_actions = board.valid_actions(agent_id)
        v = self.get_value(board, agent_id)
        for valid_action in valid_actions:
            sim = deepcopy(board)
            sim.step(agent_id, valid_action)
            action_scores.append(self.get_value(sim, agent_id))

        action_scores = [a[0][0] - min(action_scores)[0][0] + 1 for a in action_scores]
        return action_scores, v

    def forward(self, feat, adj, types):
        with torch.no_grad():
            return self.policy_network(feat, adj, types)

    def train_(self):
        self.num_train += 1
        batch = self.memory.sample(self.batch_size)
        # batch = Transition(*zip(*transitions))

        non_final_mask_feat = torch.tensor(tuple(map(lambda s: s is not None, batch.feat)))
        # non_final_mask_adj = torch.tensor(tuple(map(lambda s: s is not None, batch.adj)), torch.bool, self.device)
        non_final_next_feat = torch.cat([s for s in batch.feat if s is not None]).view(-1, self.n_nodes + self.n_agents,
                                                                                       self.feat_size)
        non_final_next_adj = torch.cat([s for s in batch.adj if s is not None]).view(-1, self.n_nodes + self.n_agents,
                                                                                     self.n_nodes + self.n_agents)

        # feat_batch = torch.cat(batch.before_feat).view(-1, self.n_nodes + self.n_agents, self.feat_size)
        # adj_batch = torch.cat(batch.before_adj).view(-1, self.n_nodes + self.n_agents, self.n_nodes + self.n_agents)
        # type_batch = torch.cat(batch.type).view(-1, self.n_nodes + self.n_agents)
        # reward_batch = torch.cat(batch.reward)
        #
        feat_batch = batch.before_feat
        adj_batch = batch.before_adj
        type_batch = batch.type
        reward_batch = batch.reward

        q = self.policy_network(feat_batch, adj_batch, type_batch).squeeze()[:, self.n_nodes:]

        y = torch.zeros([feat_batch.shape[0], self.n_agents], device=self.device)
        y[non_final_mask_feat] = self.target_network(non_final_next_feat, non_final_next_adj, type_batch)[:,
                                 self.n_nodes:].squeeze().detach() * self.gamma
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
