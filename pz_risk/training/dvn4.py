import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from collections import namedtuple

from training.utils import *

from utils import *

from training.gnn import GNN, HetroGNN
from torch_geometric.loader import DataLoader

from core.gamestate import GameState

Transition = namedtuple('Transition', ('board', 'reward', 'done'))


class GraphReplayMemory(object):

    def __init__(self, capacity, device, n_nodes, n_agents, feat_size):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.device = device
        self.n_nodes = n_nodes
        self.n_agents = n_agents
        self.feat_size = feat_size

    def push(self, data):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = data
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        loader = DataLoader(self.memory[:-1], batch_size, shuffle=True)
        return loader._get_iterator().__next__()

    def __len__(self):
        return len(self.memory)


class DVNAgent(nn.Module):
    def __init__(self, num_nodes, num_agents, node_feat_size, hidden_size, device='cuda:0'):
        super(DVNAgent, self).__init__()
        self.device = device
        self.policy_network = HetroGNN(node_feat_size, 2 * node_feat_size, hidden_size).to(device)
        self.target_network = HetroGNN(node_feat_size, 2 * node_feat_size, hidden_size).to(device)

        # self.policy_network = to_hetero(self.policy_network, data.metadata())
        # self.target_network = to_hetero(self.target_network, data.metadata())
        # self.policy_network(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
        self.memory = GraphReplayMemory(2 ** 14, device, num_nodes, num_agents, node_feat_size)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.RMSprop(self.policy_network.parameters(), lr=1e-5)

        self.batch_size = 128
        self.gamma = 0.999
        self.target_update = 20
        self.num_train = 0
        self.n_nodes = num_nodes
        self.n_agents = num_agents
        self.feat_size = node_feat_size

    def get_value_and_probs(self, board, agent_id):
        data = get_geom_from_board(board, self.n_agents, agent_id)
        x, e = self.forward(data)  # .detach().cpu().numpy()[:, self.n_nodes + agent_id]
        # print(x.shape, e.shape)
        p = None
        if self.state == GameState.Reinforce:
            p = x[data.reinforce_index].squeeze().detach().cpu().numpy()
        elif self.state == GameState.Card:
            p = [0, 1]
        elif self.state == GameState.Attack:
            p = np.insert(e[data.attack_index].squeeze().detach().cpu().numpy(), 0, 0.1)
        elif self.state == GameState.Move:
            p = e[data.move_index].squeeze().detach().cpu().numpy()
        elif self.state == GameState.Fortify:
            p = np.insert(e[data.fortify_index].squeeze().detach().cpu().numpy(), 0, 0.1)

        return x[data.value_index].item(), p

    def get_value(self, board, agent_id):
        data = get_geom_from_board(board, self.n_agents, agent_id)
        x, e = self.forward(data)  # .detach().cpu().numpy()[:, self.n_nodes + agent_id]
        # print(x.shape, e.shape)
        return x[data.node_type == 1][agent_id].item()

    def predict(self, board, agent_id):
        action_scores = []
        deterministic, valid_actions = board.valid_actions(agent_id)
        v = self.get_value(board, agent_id)
        for valid_action in valid_actions:
            sim = deepcopy(board)
            sim.step(agent_id, valid_action)
            score = 10 if len(sim.player_nodes(agent_id)) == self.n_nodes else self.get_value(sim, agent_id)
            action_scores.append(score)

        action_scores = np.exp(action_scores) / np.sum(np.exp(action_scores))
        return action_scores, v

    def forward(self, data):
        with torch.no_grad():
            return self.policy_network(data.x, data.edge_index, data.node_type,
                                       data.edge_type, data.edge_attr)

    def train_(self):
        self.num_train += 1
        batch = self.memory.sample(self.batch_size)
        non_terminal_state = torch.zeros(batch.x.shape[0])
        non_terminal_value = torch.zeros(batch.reward.shape[0])
        for i, done in enumerate(batch.done):
            done = done.item()
            if done:
                s = batch.ptr[i].item()
                e = batch.ptr[i + 1].item()
                # print(i, batch.ptr[i], batch.ptr[i + 1])
                non_terminal_state[s:e] = 1
                non_terminal_value[2 * i:2 * i + 1] = 1
        # if sum(non_terminal_state) > 1:
        #     print('asdf')
        non_terminal_state = non_terminal_state == 0
        non_terminal_value = non_terminal_value == 0
        v, p = self.policy_network(batch.last_x, batch.last_edge_index,
                                   batch.last_node_type, batch.last_edge_type, batch.last_edge_attr)
        q = v[batch.last_node_type == 1].squeeze()
        v, p = self.target_network(batch.x, batch.edge_index, batch.node_type, batch.edge_type, batch.edge_attr)
        y = torch.zeros(q.shape[0])
        y[non_terminal_value] = v[batch.node_type == 1][non_terminal_value].squeeze() * self.gamma
        y += batch.reward

        # Compute Huber loss
        loss = F.smooth_l1_loss(y, q)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_network.parameters():
            if param.grad is not None:
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
        self.memory.push(transition)
