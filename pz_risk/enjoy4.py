import os
import torch
import numpy as np

from risk_env import env
import training.utils as utils
from training.dvn7 import DVNAgent
from training.args import get_args
import wrappers

from agents.sampling import SAMPLING
from agents.value import get_future, get_attack_dist
from utils import get_geom_from_board
from agents import GreedyAgent, RandomAgent
from copy import deepcopy

from tqdm import tqdm
import datetime
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

    n_agents = args.max_agents
    node = args.max_nodes
    id = np.random.randint(n_agents)
    print('Agents: {} Node: {}, ID: {}'.format(n_agents, node, id))
    e = env(n_agent=n_agents, board_name='d_{}_{}_random'.format(node, node * 2))
    print('Agents: {} Node: {}'.format(n_agents, node))
    e = wrappers.PureGraphObservationWrapper(e)
    e = wrappers.SparseRewardWrapper(e)
    e.reset()

    feat_size = 4  # e.observation_spaces['feat'].shape[0]
    type_size = node
    hidden_size = args.hidden_size
    critic = DVNAgent(node, n_agents, feat_size, hidden_size, device)
    critic.load_state_dict(torch.load(args.dir))
    critic.eval()
    e.reset()
    state, _, _, _ = e.last()

    for i, agent_id in enumerate(e.agent_iter(max_iter=args.num_steps)):
        if agent_id != id:
            state, _, _, _ = e.last()
            task_id = state['task_id']
            env_action = SAMPLING[task_id](e.unwrapped.board, agent_id)
            # p1 = mcts.getActionProb(self.env.unwrapped.board, agent_id, None, True)
            # deterministic, valid_actions = self.env.unwrapped.board.valid_actions(agent_id)
            # action_id = np.random.choice(len(valid_actions), p=p1)
            # action = valid_actions[action_id]
        else:
            # Use Model to Gather Future State per Valid Actions
            model = critic
            # Use Model to Gather Future State per Valid Actions
            value, action, action_log_prob = model.predict(e.unwrapped.board, agent_id)
            deterministic, valid_actions = e.unwrapped.board.valid_actions(agent_id)
            env_action = valid_actions[action]
        for agent_id in e.possible_agents:
            critic_score[agent_id].append(critic.get_value(e.unwrapped.board, agent_id)[agent_id].detach().cpu().numpy())
            e.unwrapped.land_hist[agent_id].append(len(e.unwrapped.board.player_nodes(agent_id)))
            e.unwrapped.unit_hist[agent_id].append(e.unwrapped.board.player_units(agent_id))
            e.unwrapped.place_hist[agent_id].append(e.unwrapped.board.calc_units(agent_id))

        e.step(env_action)
        state, _, _, _ = e.last()
        if all(state['dones']):
            break  # state['rewards'][id]
        e.render()
        render_info()
        # print(len(e.unwrapped.board.player_nodes(id)))
    print(id, state['rewards'][id])


if __name__ == "__main__":
    main()
