import os
import torch
import random
import numpy as np

from risk_env import env
import training.utils as utils
from training.dvn import DVNAgent
from training.arguments import get_args
from wrappers import GraphObservationWrapper

from agents.value import get_future, get_attack_dist, manual_value
from copy import deepcopy

from utils import get_feat_adj_from_board
from tqdm import tqdm

from agents.sampling import SAMPLING

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


def main():
    args = get_args()

    torch.manual_seed(args.seed + 1000)
    torch.cuda.manual_seed_all(args.seed + 1000)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    e = env(n_agent=6, board_name='world')
    e = GraphObservationWrapper(e)
    e.reset()
    _, _, _, info = e.last()
    n_nodes = info['nodes']
    n_agents = info['agents']

    feat_size = e.observation_spaces['feat'].shape[0]
    hidden_size = 20

    critic = DVNAgent(n_nodes, n_agents, feat_size, hidden_size)
    critic.load_state_dict(torch.load(args.dir))
    critic.eval()
    e.reset()
    state, _, _, _ = e.last()
    max_episode = 100
    result = []
    for _ in tqdm(range(max_episode)):
        e.reset()
        for agent_id in e.agent_iter(max_iter=20000):
            state, _, _, info = e.last()
            if len(e.unwrapped.board.player_nodes(0)) == n_nodes:
                result.append(1)
                break
            elif len(e.unwrapped.board.player_nodes(0)) == 0:
                result.append(-1)
                break
            # make an action based on epsilon greedy action
            if agent_id != 0 or True:
                task_id = state['task_id']
                action = SAMPLING[task_id](e.unwrapped.board, agent_id)
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
                            left = get_future(dist, mode='most')
                            sim.step(agent_id, valid_action, left)
                        else:
                            sim.step(agent_id, valid_action)
                    sim_feat, sim_adj = get_feat_adj_from_board(sim, agent_id, e.unwrapped.n_agents, e.unwrapped.n_grps)
                    sim_feat = torch.tensor(sim_feat, dtype=torch.float32, device=device).reshape(-1,
                                                                                                  n_nodes + n_agents,
                                                                                                  feat_size)
                    sim_adj = torch.tensor(sim_adj, dtype=torch.float32, device=device).reshape(-1, n_nodes + n_agents,
                                                                                                n_nodes + n_agents)
                    action_scores.append(critic(sim_feat, sim_adj).detach().cpu().numpy()[:, n_nodes + agent_id])
                action = valid_actions[np.argmax(action_scores)]

            e.step(action)

    print(sum(result), sum([r for r in result if r > 0]), sum([r for r in result if r < 0]), max_episode - len(result))


if __name__ == "__main__":
    main()
