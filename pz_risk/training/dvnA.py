import numpy
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from collections import namedtuple

from training.utils import *

from utils import *

from training.gnn import HATGNN
from torch_geometric.loader import DataLoader

from core.gamestate import GameState

from training.distributions import *
from tqdm import tqdm
import wandb


class GraphReplayMemory(object):

    def __init__(self, capacity, num_envs, device):
        self.capacity = capacity
        self.memory = [[] for _ in range(num_envs)]
        self.position = [0 for _ in range(num_envs)]
        self.device = device

    def compute_returns(self, next_value, gamma):
        for i, memory in enumerate(self.memory):
            memory[-1].returns = next_value[i] * gamma * memory[-1].mask + memory[-1].reward
            advantages = torch.zeros([len(memory), memory[0].reward.shape[0]], device=self.device)
            for step in reversed(range(len(memory) - 1)):
                memory[step].returns = memory[step + 1].returns * gamma * memory[step].mask \
                                            + memory[step].reward
                advantages[step] = memory[step].returns - memory[step].value
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
            for step in range(len(memory)):
                memory[step].advantages = advantages[step]

    def push(self, i, data):
        """Saves a transition."""
        if len(self.memory[i]) < self.capacity:
            self.memory[i].append(None)
        self.memory[i][self.position[i]] = data
        self.position[i] = (self.position[i] + 1) % self.capacity

    def get_loader(self, batch_size):
        full_mem = []
        for mem in self.memory:
            full_mem += mem
        return DataLoader(full_mem, batch_size, shuffle=True)

    def sample(self, batch_size):
        full_mem = []
        for mem in self.memory:
            full_mem += mem
        loader = DataLoader(full_mem, batch_size, shuffle=True)
        return loader._get_iterator().__next__()

    def __len__(self):
        return len(self.memory[0])


class DVNAgent(nn.Module):
    def __init__(self, num_nodes, num_agents, node_feat_size, hidden_size, num_envs, device='cuda:0'):
        super(DVNAgent, self).__init__()
        self.device = device
        self.policy_network = HATGNN(node_feat_size, 0, hidden_size, device).to(device)
        self.memory = GraphReplayMemory(2 ** 14, num_envs, device)

        self.optimizer = optim.RMSprop(self.policy_network.parameters(), lr=1e-4)

        self.batch_size = 128
        self.gamma = 0.999
        self.num_train = 0
        self.n_nodes = num_nodes
        self.n_agents = num_agents
        self.feat_size = node_feat_size

    def _get_value_and_logits_batch(self, data, advantage=False):
        p = []
        a = []
        v = []

        x, e = self.forward_grad(data)
        x_pointer = 0
        e_pointer = 0
        for i in range(len(data.to_data_list())):
            datum = data[i]
            # x, e = self.forward_grad(datum)
            v.append(x[x_pointer: x_pointer + datum.x.shape[0]][datum.value_index].squeeze(1))
            if advantage:
                a.append(datum.advantages[datum.agent_id])
            if datum.task_id == GameState.Reinforce:
                p.append(x[x_pointer: x_pointer + datum.x.shape[0]][datum.reinforce_index].squeeze(1))
            elif datum.task_id == GameState.Card:
                p.append(torch.tensor([0, 1], device=self.device))
            elif datum.task_id == GameState.Attack:
                p.append(torch.cat([torch.tensor([0.1], device=self.device),
                                    e[e_pointer: e_pointer + datum.e.shape[0]][datum.attack_index].squeeze(1)]))
            elif datum.task_id == GameState.Move:
                p.append(torch.sigmoid(e[e_pointer: e_pointer + datum.e.shape[0]][datum.move_index].squeeze(1)))
            elif datum.task_id == GameState.Fortify:
                p.append(torch.cat([torch.tensor([0.1], device=self.device),
                                    e[e_pointer: e_pointer + datum.e.shape[0]][datum.fortify_index].squeeze(1)]))
            else:
                p.append(torch.tensor([0], device=self.device))
        # p = torch.cat(p)
        if advantage:
            v = torch.cat(v)
            a = torch.cat(a)
            return v, p, a
        return v, p

    def _get_value_and_logits(self, data):
        x, e = self.forward(data)
        p = None
        if data.task_id == GameState.Reinforce:
            p = x[data.reinforce_index].squeeze(1)
        elif data.task_id == GameState.Card:
            p = [0, 1]
        elif data.task_id == GameState.Attack:
            p = torch.cat([torch.tensor([0.1]), e[data.attack_index].squeeze(1)])
        elif data.task_id == GameState.Move:
            p = torch.sigmoid(e[data.move_index].squeeze(1))
        elif data.task_id == GameState.Fortify:
            p = torch.cat([torch.tensor([0.1]), e[data.fortify_index].squeeze(1)])
        else:
            print(data.task_id)
        return x[data.value_index].squeeze(1), p

    def get_value_and_logits(self, board, agent_id):
        data = get_geom_from_board(board, self.n_agents, agent_id)
        return self._get_value_and_logits(data)

    def get_value_batch(self, data):
        x, e = self.forward(data)
        x_pointer = 0
        v = []
        for i in range(len(data.to_data_list())):
            datum = data[i]
            v.append(x[x_pointer: x_pointer + datum.x.shape[0]][datum.value_index].squeeze(1))

        return v

    def get_value(self, data):
        x, e = self.forward(data)  # .detach().cpu().numpy()[:, self.n_nodes + agent_id]
        # print(x.shape, e.shape)
        return x[data.value_index].squeeze(1)

    def evaluate_action(self, data):
        value, logits_array, adv = self._get_value_and_logits_batch(data, advantage=True)
        action_log_prob = []
        dist_entropy = []
        for logits, action, task in zip(logits_array, data.action, data.task_id):
            dist = FixedCategorical(logits=logits)
            action_log_prob.append(dist.log_prob(action).unsqueeze(0))
            dist_entropy.append(dist.entropy().mean().unsqueeze(0))
        action_log_prob = torch.cat(action_log_prob)
        dist_entropy = torch.cat(dist_entropy).mean()
        return value, action_log_prob, dist_entropy, adv

    def predict(self, data):
        value_array, logits_array = self._get_value_and_logits_batch(data)
        actions = []
        action_log_probs = []
        for logits in logits_array:
            dist = FixedCategorical(logits=logits)
            action = dist.sample()
            actions.append(action)
            action_log_probs.append(dist.log_prob(action))
        return value_array, actions, action_log_probs

    def forward_grad(self, data):
        return self.policy_network(data.x, data.edge_index, data.node_type, data.edge_type, data.edge_attr)

    def forward(self, data):
        with torch.no_grad():
            return self.policy_network(data.x, data.edge_index, data.node_type, data.edge_type, data.edge_attr)

    def train_(self, train_epochs):
        self.num_train += 1
        clip_param = 0.2
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        t = tqdm(range(train_epochs), desc='Train Epoch')
        for _ in t:
            i = _ + 1
            loader = self.memory.get_loader(self.batch_size)
            for batch in loader:
                value, action_log_prob, dist_entropy, adv = self.evaluate_action(batch)
                ratio = torch.exp(action_log_prob - batch.action_log_prob)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv
                action_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * (batch.returns - batch.value).pow(2).mean()
                self.optimizer.zero_grad()
                loss = (value_loss * 0.5 + action_loss - dist_entropy * 0.01)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1)

                self.optimizer.step()
                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                t.set_postfix(VLoss=value_loss.item(), ALoss=action_loss.item(), DLoss=dist_entropy.item(),
                              LOSS=loss.item())
                n = ''  # {}'.format(self.n_nodes)
                # wandb.log({
                #     'LOSS' + n: loss.item(),
                #     'Dist Entropy' + n: dist_entropy.item(),
                #     'Value Loss' + n: value_loss.item(),
                #     'Action Loss' + n: action_loss.item(),
                # })
        num_updates = train_epochs * self.batch_size

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

    def train_start(self):
        return len(self.memory) > self.batch_size  # Boolean
