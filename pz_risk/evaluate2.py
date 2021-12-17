import os
import torch
import numpy as np

from risk_env import env
import training.utils as utils
from training.dvn2 import DVNAgent
from training.args import get_args
import wrappers

from agents.sampling import SAMPLING
from agents.value import get_future, get_attack_dist
from utils import get_feat_adj_from_board, get_feat_adj_type_from_board
from agents import GreedyAgent, RandomAgent
from copy import deepcopy

from tqdm import tqdm
import datetime

from multiprocessing import Pool
from training.mcts import MCTS

class Coach:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.n_nodes = -1
        self.n_agents = -1
        self.env = None
        self.feat_size = -1
        self.type_size = -1
        self.critic = None
        self.last_critic = None

        self.epsilon = 1.0
        self.epsilon_min = 0.0005
        self.decay_rate = 0.05

        self.episode = 0
        self.last_save_episode = 0

        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.save_path = self.args.save_path + '{}_{}_{}_{}_{}.pt'.format(timestamp, args.max_agents, args.max_nodes,
                                                                          args.seed, '{}')
        self.agents = range(2, self.args.max_agents)
        self.nodes = range(10, self.args.max_nodes)
        self.exec_pool = Pool()
        self.eval_pool = Pool()

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['exec_pool']
        del self_dict['eval_pool']
        return self_dict

    def execute(self):

        for agent in self.agents:
            for node in self.nodes:
                if node >= agent * 2:
                    # self.episode = episode
                    self.n_agents = agent
                    self.n_nodes = node
                    self.initialize()
                    self.evaluate()
                    self.finalize()

    def initialize(self):
        # self.n_agents = range(2, self.args.max_agents)
        # self.n_nodes = range(self.args.max_agents + 1, self.args.max_nodes)
        self.env = env(n_agent=self.n_agents, board_name='d_{}_{}_random'.format(self.n_nodes, self.n_nodes * 2))
        print('Agents: {} Node: {}'.format(self.n_agents, self.n_nodes))
        self.env = wrappers.PureGraphObservationWrapper(self.env)
        self.env = wrappers.SparseRewardWrapper(self.env)
        self.env.reset()

        self.feat_size = self.env.observation_spaces['feat'].shape[0]
        self.type_size = self.n_nodes
        hidden_size = self.args.hidden_size
        self.critic = DVNAgent(self.n_nodes, self.n_agents, self.type_size, self.feat_size, hidden_size, self.device)

        self.critic.load_state_dict(torch.load(self.args.dir))

    def eval(self, ids):
        self.env.reset()
        j, id = ids
        mcts = MCTS( self.n_agents, self.args)
        for i, agent_id in enumerate(self.env.agent_iter(max_iter=self.args.num_steps)):
            if agent_id != id:
                state, _, _, _ = self.env.last()
                task_id = state['task_id']
                action = SAMPLING[task_id](self.env.unwrapped.board, agent_id)
                # p1 = mcts.getActionProb(self.env.unwrapped.board, agent_id, None, True)
                # deterministic, valid_actions = self.env.unwrapped.board.valid_actions(agent_id)
                # action_id = np.random.choice(len(valid_actions), p=p1)
                # action = valid_actions[action_id]
            else:
                model = self.critic
                # Use Model to Gather Future State per Valid Actions
                action_scores = []
                deterministic, valid_actions = self.env.unwrapped.board.valid_actions(agent_id)
                for valid_action in valid_actions:
                    sim = deepcopy(self.env.unwrapped.board)
                    sim.step(agent_id, valid_action)
                    sim_feat, sim_adj, sim_type = get_feat_adj_type_from_board(sim, self.env.unwrapped.n_agents)
                    sim_feat = torch.tensor(sim_feat, dtype=torch.float32, device=self.device).reshape(-1,
                                                                                                       self.n_nodes + self.n_agents,
                                                                                                       self.feat_size)
                    sim_adj = torch.tensor(sim_adj, dtype=torch.float32, device=self.device).reshape(-1,
                                                                                                     self.n_nodes + self.n_agents,
                                                                                                     self.n_nodes + self.n_agents)
                    sim_type = torch.tensor(sim_type, dtype=torch.float32, device=self.device).reshape(-1,
                                                                                                       self.n_nodes + self.n_agents)
                    # if len(sim.player_nodes(agent_id)) == self.n_nodes:
                        # action_scores.append([[10000]])
                    # else:
                    action_scores.append(model(sim_feat, sim_adj, sim_type).detach().cpu().numpy()[:, self.n_nodes + agent_id])
                action_scores = [a[0][0] for a in action_scores]
                action_scores -= min(action_scores)
                action_scores += 1
                p = [float(s) / sum(action_scores) for s in action_scores]
                # print([a*len(p) for a in p])
                action_id = np.random.choice(len(action_scores), p=p)
                # action_id = np.argmax(action_scores)
                action = valid_actions[action_id]
            self.env.step(action)
            state, _, _, _ = self.env.last()
            if all(state['dones']):
                return state['rewards'][id]
        return 0

    def evaluate(self):
        self.critic.eval()
        ids = [i for i in range(self.n_agents)] * self.args.eval_iter
        t = self.eval_pool.map(self.eval, enumerate(ids))
        w = len([tt for tt in t if tt > 0])
        l = len([tt for tt in t if tt < 0])
        print(w, l, len(t) - w - l, (w - l) / len(t))

    def finalize(self):
        self.env.close()


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    coach = Coach(args, device)

    coach.execute()


if __name__ == "__main__":
    main()
