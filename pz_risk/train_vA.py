import random

import torch
import numpy as np

from risk_env import env
from training.dvnA import DVNAgent
from training.args import get_args
import wrappers

from agents.sampling import SAMPLING

from tqdm import tqdm
import datetime

# from multiprocessing import Pool as CPool
# from torch.multiprocessing import Pool, set_start_method

from training.mcts import MCTS
from typing import Union
import wandb
import utils
from torch_geometric.loader import DataLoader


def make_batch(data, batch_size, device):
    loader = DataLoader(data, batch_size, shuffle=False)
    return loader._get_iterator().__next__().to(device)

class Coach:
    def __init__(self, args, device):
        # wandb.init(project='Risk on Graph')
        self.v_env = None
        self.args = args
        self.device = device
        self.n_nodes = -1
        self.n_agents = -1
        self.edge_p = -1
        self.n_units = -1
        self.feat_size = -1
        self.type_size = -1
        self.critic: Union[DVNAgent, None] = None
        self.last_critic: Union[DVNAgent, None] = None
        self.n = self.args.num_processes
        self.epsilon = 1.0
        self.epsilon_min = 0.0005
        self.decay_rate = 0.05

        # self.mcts = MCTS(self.n_agents, self.args)

        self.episode = 0
        self.last_save_episode = 0
        self.hidden_size = self.args.hidden_size
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.save_path = self.args.save_path + '{}_{}_{}_{}_{}.pt'.format(args.max_agents, args.max_nodes,
                                                                          args.hidden_size, args.seed, '{}')
        # if self.args.cuda:
        #     try:
        #         set_start_method('spawn')
        #     except RuntimeError:
        #         pass
        #     self.exec_pool = Pool(args.num_processes)
        #     self.eval_pool = Pool(args.eval_iter * args.max_agents)
        # else:
        #     self.exec_pool = CPool(args.num_processes)
        #     self.eval_pool = CPool(args.eval_iter * args.max_agents)

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['exec_pool']
        del self_dict['eval_pool']
        return self_dict

    def execute(self):
        t = tqdm(range(self.args.load, self.args.max_episode + 1), 'Self Play')
        for episode in t:
            self.episode = episode
            self.n_agents = np.random.randint(2, self.args.max_agents)
            f = self.args.max_nodes - self.n_agents * 2
            self.n_nodes = np.random.choice(range(self.n_agents * 2, self.args.max_nodes),
                                            p=[(f - i) / sum(range(f + 1)) for i in range(f)])
            self.edge_p = np.random.random() * 0.2 + 0.4
            self.n_units = np.random.randint(2 * self.n_nodes, 3 * self.n_nodes)
            t.set_postfix(Agents=self.n_agents, Nodes=self.n_nodes, P=self.edge_p)

            # print('Init...')
            self.initialize()
            # print('Exe...')
            self.exe()
            # print('Train...')
            self.train()
            # print('Eval...')
            self.evaluate()
            self.finalize()

    def initialize(self):
        envs = []
        for i in range(self.n):
            e = env(n_agent=self.n_agents,
                    board_name='d_{}_{}_random'.format(self.n_nodes, self.n_units),
                    edge_p=self.edge_p, id=i)
            e = wrappers.GeometricObservationWrapper(e)
            e = wrappers.SparseRewardWrapper(e)
            envs.append(e)

        self.v_env = wrappers.VectorizeWrapper(envs)
        self.v_env.reset()
        self.feat_size = 4
        self.type_size = self.n_nodes
        self.critic = DVNAgent(self.n_nodes, self.n_agents, self.feat_size, self.hidden_size, self.n, self.device)
        self.last_critic = DVNAgent(self.n_nodes, self.n_agents, self.feat_size, self.hidden_size, self.n, self.device)
        if self.episode == 0:
            torch.save(self.critic.state_dict(), self.save_path.format(self.episode))
            self.last_save_episode = self.episode

        self.critic.load_state_dict(torch.load(self.save_path.format(self.last_save_episode)))
        self.last_critic.load_state_dict(self.critic.state_dict())

    def exe(self):
        if self.args.cuda:
            # self.device = 'cuda:{}'.format(np.random.randint(7))
            self.critic.to(self.device)
        self.v_env.reset()
        last_players = []
        for i, agent_ids in enumerate(self.v_env.agent_iter(max_iter=self.args.num_steps)):
            states, _, _, info = self.v_env.last()
            last_players = agent_ids
            self.critic.eval()
            # data = self.v_env.get_data(agent_ids, self.n_agents)
            # data = [d.to(self.device) for d in data]
            data = [state['data'] for state in states]
            batch = make_batch(data, len(data), self.device)
            value, action, action_log_prob = self.critic.predict(batch)
            valid_actions = self.v_env.valid_actions(agent_ids)
            env_actions = [va[a] for va, a in zip(valid_actions, action)]

            self.v_env.step(env_actions)
            next_states, _, _, info = self.v_env.last()
            reward = [s['rewards'] for s in next_states]
            done = [s['dones'] for s in next_states]
            if i == self.args.num_steps - 1:
                rew = [self.v_env.reward(i, True) for i in range(self.n_agents)]
                reward = rew

            for j in range(self.n):
                data = states[j]['data']
                data.action = action[j]
                data.action_log_prob = action_log_prob[j]
                data.value = value[j]
                next_data = next_states[j]['data']
                data.reward = torch.tensor(reward[j], device=self.device)
                data.done = torch.tensor(all(done[j]), device=self.device)
                data.next_x = next_data.x
                data.next_edge_index = next_data.edge_index
                data.next_edge_attr = next_data.edge_attr
                data.next_edge_type = next_data.edge_type
                data.next_node_type = next_data.node_type
                data.mask = torch.zeros(len(reward[j]), device=self.device) if all(done[j]) else torch.ones(len(reward[j]), device=self.device)
                data.to(self.device)
                self.critic.memory.push(self.v_env.envs[j].id, data)

            for j in range(self.n):
                if all(done[j]):
                    self.n -= 1
                    self.v_env.envs = [e for k, e in enumerate(self.v_env.envs) if k != j]
            if len(self.v_env.envs) == 0:
                break
        data = self.v_env.get_data(last_players, self.n_agents)
        batch = make_batch(data, len(data), self.device)
        final_value = self.critic.get_value_batch(batch)
        self.critic.memory.compute_returns(final_value, self.args.gamma)

    def train(self):
        # self.critic.to('cuda:4')
        self.critic.train()
        # if self.critic.train_start():
        loss = self.critic.train_(self.args.train_epoch)
        print('Trained: ', loss)
        # return True
        # return False

    def eval(self, ids):
        j, id = ids
        self.v_env.reset()
        if self.args.cuda:
            self.device = 'cuda:{}'.format(np.random.randint(7))
            self.critic.to(self.device)
            self.last_critic.to(self.device)
        # id = np.random.randint(self.n_agents)
        out = []
        for i, agent_id in enumerate(self.v_env.agent_iter(max_iter=self.args.num_steps)):
            model = self.critic if agent_id == id else self.last_critic
            # Use Model to Gather Future State per Valid Actions
            data = self.v_env.get_data(agent_id, self.n_agents)
            value, action, action_log_prob = model.predict(data)
            valid_actions = self.v_env.valid_actions(agent_id)
            env_actions = [va[a] for va, a in zip(valid_actions, action)]

            self.v_env.step(env_actions)
            state, _, _, _ = self.v_env.last()
            if all(state['dones']):
                return state['rewards'][id]
        return 0

    def evaluate(self):
        self.critic.eval()
        self.last_critic.eval()
        w = 0
        y = 0
        ids = [i for i in range(self.n_agents)] * self.args.eval_iter
        t = self.eval(ids)
        for j in range(self.args.eval_iter):
            if j > self.args.eval_iter // 2 and False:
                y += t[j]
            else:
                w += t[j]
        if w > 1:
            print(w, y, 'Accept')
            torch.save(self.critic.state_dict(), self.save_path.format(self.episode))
            self.last_save_episode = self.episode
        else:
            print(w, y, 'Reject')
            self.critic.load_state_dict(torch.load(self.save_path.format(self.last_save_episode)))

    def finalize(self):
        self.env.close()
        #
        # if episode + 1 % self.args.save_interval == 0 and episode // self.args.save_interval > 0:
        #     torch.save(self.critic.state_dict(),
        #                os.path.join(self.args.save_path, str(episode // self.args.save_interval) + ".pt"))


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # torch.set_num_threads(1)
    i = 6
    device = torch.device("cuda:{}".format(i) if args.cuda else "cpu")

    coach = Coach(args, device)

    coach.execute()


if __name__ == "__main__":
    main()
