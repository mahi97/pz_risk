import numpy
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from collections import namedtuple

from training.utils import *

from utils import *

from training.gnn import HEATGNN
from torch_geometric.loader import DataLoader

from core.gamestate import GameState

from training.distributions import *
from tqdm import tqdm
import wandb

class GraphReplayMemory(object):

    def __init__(self, capacity, device):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.device = device

    def compute_returns(self, next_value, gamma):
        self.memory[-1].returns = next_value * gamma * self.memory[-1].mask + self.memory[-1].reward
        advantages = torch.zeros([len(self.memory), self.memory[0].reward.shape[0]])
        for step in reversed(range(len(self.memory) - 1)):
            self.memory[step].returns = self.memory[step + 1].returns * gamma * self.memory[step].mask \
                                        + self.memory[step].reward
            advantages[step] = self.memory[step].returns - self.memory[step].value
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        for step in range(len(self.memory)):
            self.memory[step].advantages = advantages[step]

    def push(self, data):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = data
        self.position = (self.position + 1) % self.capacity

    def get_loader(self, batch_size):
        return DataLoader(self.memory, batch_size, shuffle=True)

    def sample(self, batch_size):
        loader = DataLoader(self.memory  # [:-1]
                            , batch_size, shuffle=True)
        return loader._get_iterator().__next__()

    def __len__(self):
        return len(self.memory)


class DVNAgent(nn.Module):
    def __init__(self, num_nodes, num_agents, node_feat_size, hidden_size, device='cuda:0'):
        super(DVNAgent, self).__init__()
        self.device = device
        self.policy_network = HEATGNN(node_feat_size, 0, hidden_size).to(device)

        self.memory = GraphReplayMemory(2 ** 14, device)

        self.optimizer = optim.RMSprop(self.policy_network.parameters(), lr=1e-4)

        self.batch_size = 128
        self.gamma = 0.999
        self.num_train = 0
        self.n_nodes = num_nodes
        self.n_agents = num_agents
        self.feat_size = node_feat_size

    def _get_value_and_logits_batch(self, data):
        p = []
        v = []
        a = []
        for i in range(len(data.to_data_list())):
            datum = data[i]
            x, e = self.forward_grad(datum)
            v.append(x[datum.value_index].squeeze(1))
            a.append(datum.advantages[datum.agent_id])
            if datum.task_id == GameState.Reinforce:
                p.append(x[datum.reinforce_index].squeeze(1))
            elif datum.task_id == GameState.Card:
                p.append(torch.tensor([0, 1]))
            elif datum.task_id == GameState.Attack:
                p.append(torch.cat([torch.tensor([0.1]), e[datum.attack_index].squeeze(1)]))
            elif datum.task_id == GameState.Move:
                p.append(torch.sigmoid(e[datum.move_index].squeeze(1)))
            elif datum.task_id == GameState.Fortify:
                p.append(torch.cat([torch.tensor([0.1]), e[datum.fortify_index].squeeze(1)]))
            else:
                p.append(torch.tensor([0]))
        # p = torch.cat(p)
        v = torch.cat(v)
        a = torch.cat(a)
        return v, p, a

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

    def get_value(self, board, agent_id):
        data = get_geom_from_board(board, self.n_agents, agent_id)
        x, e = self.forward(data)  # .detach().cpu().numpy()[:, self.n_nodes + agent_id]
        # print(x.shape, e.shape)
        return x[data.value_index].squeeze(1)

    def evaluate_action(self, data):
        value, logits_array, adv = self._get_value_and_logits_batch(data)
        action_log_prob = []
        dist_entropy = []
        for logits, action, task in zip(logits_array, data.action, data.task_id):
            dist = FixedCategorical(logits=logits)
            action_log_prob.append(dist.log_prob(action).unsqueeze(0))
            dist_entropy.append(dist.entropy().mean().unsqueeze(0))
        action_log_prob = torch.cat(action_log_prob)
        dist_entropy = torch.cat(dist_entropy).mean()
        return value, action_log_prob, dist_entropy, adv

    def predict(self, board, agent_id):
        # action_scores = []
        # deterministic, valid_actions = board.valid_actions(agent_id)
        value, logits = self.get_value_and_logits(board, agent_id)
        if logits.dim() == 0:
            print(logits)
        dist = FixedCategorical(logits=logits)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return value, action, action_log_prob
        # if board.state == GameState.Move:
        #     a = int(probs * (len(valid_actions)-1))
        #     probs = [(0.9 if a == i else 0.1) for i in valid_actions]
        # probs = np.exp(probs) / np.sum(np.exp(probs))
        # if isinstance(probs, numpy.float32):
        #     probs = [probs]
        # return probs, v

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
                t.set_postfix(VLoss=value_loss.item(), ALoss=action_loss.item(), DLoss=dist_entropy.item(), LOSS=loss.item())
                n = ''  #  {}'.format(self.n_nodes)
                wandb.log({
                    'LOSS' + n: loss.item(),
                    'Dist Entropy' + n: dist_entropy.item(),
                    'Value Loss' + n: value_loss.item(),
                    'Action Loss' + n: action_loss.item(),
                })
        num_updates = train_epochs * self.batch_size

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

    def train_start(self):
        return len(self.memory) > self.batch_size  # Boolean

    def save_memory(self, transition):
        self.memory.push(transition)
