import os
import torch
import random
import numpy as np

from risk_env import env
import training.utils as utils
from training.dvn2 import DVNAgent
from training.args import get_args
from wrappers import PureGraphObservationWrapper, GraphObservationWrapper, DenseRewardWrapper, SparseRewardWrapper

from agents.value import get_future, get_attack_dist, manual_value
from copy import deepcopy

from utils import get_feat_adj_from_board, get_feat_adj_type_from_board

import matplotlib.pyplot as plt

from agents.sampling import SAMPLING
from training.mcts import MCTS


COLORS = [
    'tab:red',
    'tab:blue',
    'tab:green',
    'tab:purple',
    'tab:pink',
    'tab:cyan',
]

critic_score = {a: [] for a in range(6)}
value_score = {a: [] for a in range(6)}


def render_info(mode="human"):
    global critic_score
    fig = plt.figure(2, figsize=(10, 5))
    plt.clf()

    ax1 = fig.add_subplot(121)
    for a in range(6):
        ax1.plot(critic_score[a], COLORS[a])
        ax1.set_title('Critic Score')
    ax2 = fig.add_subplot(122)
    for a in range(6):
        ax2.plot(value_score[a], COLORS[a])
        ax2.set_title('Value Score')
    plt.pause(0.001)


def main():
    args = get_args()

    torch.manual_seed(args.seed + 1000)
    torch.cuda.manual_seed_all(args.seed + 1000)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    n_agents = 2 # np.random.randint(2, 6)
    node = 11 #np.random.randint(3, 20 // n_agents) * n_agents
    id = np.random.randint(n_agents)
    print('Agents: {} Node: {}, ID: {}'.format(n_agents, node, id))
    e = env(n_agent=n_agents, board_name='d_{}_{}_random'.format(node, node * 2))
    e = PureGraphObservationWrapper(e)
    e = SparseRewardWrapper(e)
    e.reset()
    _, _, _, info = e.last()
    n_nodes = info['nodes']
    mcts = MCTS(args)
    feat_size = e.observation_spaces['feat'].shape[0]
    hidden_size = 512
    type_size = node
    critic = DVNAgent(n_nodes, n_agents, type_size, feat_size, hidden_size, device)
    save_path = './mini_6/'
    load = 80
    critic.load_state_dict(torch.load(args.dir))
    critic.eval()
    e.reset()
    state, _, _, _ = e.last()

    for agent_id in e.agent_iter(max_iter=1000):
        state, _, _, info = e.last()
        feat = torch.tensor(state['feat'], dtype=torch.float32, device=device).reshape(-1, n_nodes + n_agents,
                                                                                       feat_size)
        adj = torch.tensor(state['adj'], dtype=torch.float32, device=device).reshape(-1, n_nodes + n_agents,
                                                                                     n_nodes + n_agents)
        types = torch.tensor(state['type'], dtype=torch.float32, device=device).reshape(-1, n_nodes + n_agents)
        for a in e.possible_agents:
            e.unwrapped.land_hist[a].append(len(e.unwrapped.board.player_nodes(a)))
            e.unwrapped.unit_hist[a].append(e.unwrapped.board.player_units(a))
            e.unwrapped.place_hist[a].append(e.unwrapped.board.calc_units(a))
            critic_score[a].append(critic(feat, adj, types).detach().cpu().numpy()[:, n_nodes + a, 0][0])
            value_score[a].append(manual_value(e.unwrapped.board, a))
        # p2 = mcts.getActionProb(e.unwrapped.board, agent_id, critic, False)
        # make an action based on epsilon greedy action
        if agent_id != id:
            p1 = mcts.getActionProb(e.unwrapped.board, agent_id, critic, True)
            deterministic, valid_actions = e.unwrapped.board.valid_actions(agent_id)
            # task_id = state['task_id']
            # action = SAMPLING[task_id](e.unwrapped.board, agent_id)
            action_id = np.random.choice(len(valid_actions), p=p1)
            # action_id = np.argmax(action_scores)
            action = valid_actions[action_id]
        else:
            # Use Model to Gather Future State per Valid Actions
            action_scores = []
            deterministic, valid_actions = e.unwrapped.board.valid_actions(agent_id)
            for valid_action in valid_actions:
                sim = deepcopy(e.unwrapped.board)
                if deterministic:
                    sim.step(agent_id, valid_action)
                else:
                    dist = get_attack_dist(e.unwrapped.board, valid_action)
                    if len(dist):  # TODO: Change to sampling
                        left = get_future(dist, mode='two', risk=0.2)
                        sim.step(agent_id, valid_action, left)
                    else:
                        sim.step(agent_id, valid_action)
                # sim_feat, sim_adj = get_feat_adj_from_board(sim, agent_id, e.unwrapped.n_agents, e.unwrapped.n_grps)
                sim_feat, sim_adj, sim_type = get_feat_adj_type_from_board(sim, e.unwrapped.n_agents)
                sim_feat = torch.tensor(sim_feat, dtype=torch.float32, device=device).reshape(-1,
                                                                                              n_nodes + n_agents,
                                                                                              feat_size)
                sim_adj = torch.tensor(sim_adj, dtype=torch.float32, device=device).reshape(-1, n_nodes + n_agents,
                                                                                            n_nodes + n_agents)
                sim_type = torch.tensor(sim_type, dtype=torch.float32, device=device).reshape(-1, n_nodes + n_agents)
                if len(sim.player_nodes(agent_id)) == n_nodes:
                    action_scores.append([[10000]])
                else:
                    action_scores.append(
                        critic(sim_feat, sim_adj, sim_type).detach().cpu().numpy()[:, n_nodes + agent_id])
            action_scores = [a[0][0] for a in action_scores]
            action_scores -= min(action_scores)
            action_scores += 1

            action_id = np.random.choice(len(action_scores), p=[float(s) / sum(action_scores) for s in action_scores])
            # action_id = np.argmax(action_scores)
            action = valid_actions[action_id]
        e.step(action)
        state, _, _, _ = e.last()
        if all(state['dones']):
            break
        e.render()
        render_info()
        # print(len(e.unwrapped.board.player_nodes(id)))
    print(id, state['rewards'][id])


if __name__ == "__main__":
    main()
