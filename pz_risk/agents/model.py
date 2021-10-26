import os
import numpy as np

from core.board import Board
from agents.base import BaseAgent

import torch
from copy import deepcopy
from agents.value import get_future, get_attack_dist
from utils import get_feat_adj_from_board
from training.dvn import DVNAgent


class ModelAgent(BaseAgent):
    def __init__(self, player_id, device='cuda:0'):
        super(ModelAgent, self).__init__()
        self.player_id = player_id
        self.device = device
        feat_size = 14  # e.observation_spaces['feat'].shape[0]
        hidden_size = 20

        self.critic = DVNAgent(feat_size, hidden_size)
        save_path = './trained_models4/'
        load = 8
        self.critic.load_state_dict(torch.load(os.path.join(save_path, str(load) + ".pt")))
        self.critic.eval()

        # feat = torch.tensor(state['feat'], dtype=torch.float32, device=device).reshape(-1, 48, feat_size)
        # adj = torch.tensor(state['adj'], dtype=torch.float32, device=device).reshape(-1, 48, 48)

    def reset(self):
        pass

    def act(self, state: Board):
        action_scores = []
        deterministic, valid_actions = state.valid_actions(self.player_id)
        for valid_action in valid_actions:
            sim = deepcopy(state)
            if deterministic:
                sim.step(self.player_id, valid_action)
            else:
                dist = get_attack_dist(state, valid_action)
                if len(dist):  # TODO: Change to sampling
                    left = get_future(dist, mode='most')
                    sim.step(self.player_id, valid_action, left)
                else:
                    sim.step(self.player_id, valid_action)
            sim_feat, sim_adj = get_feat_adj_from_board(sim, self.player_id, 6, 6)
            sim_feat = torch.tensor(sim_feat, dtype=torch.float32, device=self.device).reshape(-1, 48, 14)
            sim_adj = torch.tensor(sim_adj, dtype=torch.float32, device=self.device).reshape(-1, 48, 48)
            action_scores.append(self.critic(sim_feat, sim_adj).detach().cpu().numpy()[:, 42 + self.player_id])
        action = valid_actions[np.argmax(action_scores)]
        return action
