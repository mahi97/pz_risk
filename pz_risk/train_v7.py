import torch
import numpy as np

from risk_env import env
from training.dvn6 import DVNAgent
from training.args import get_args
import wrappers

from agents.sampling import SAMPLING

from tqdm import tqdm
import datetime

from multiprocessing import Pool as CPool

from torch.multiprocessing import Pool, set_start_method
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

        self.mcts = MCTS(self.n_agents, self.args)

        self.episode = 0
        self.last_save_episode = 0
        self.hidden_size = self.args.hidden_size
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.save_path = self.args.save_path + '{}_{}_{}_{}_{}.pt'.format(args.max_agents, args.max_nodes,
                                                                          args.hidden_size, args.seed, '{}')
        if self.args.cuda:
            try:
                set_start_method('spawn')
            except RuntimeError:
                pass
            self.exec_pool = Pool(args.num_processes)
            self.eval_pool = Pool(args.eval_iter * args.max_agents)
        else:
            self.exec_pool = CPool(args.num_processes)
            self.eval_pool = CPool(args.eval_iter * args.max_agents)

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
            self.n_nodes = np.random.randint(self.n_agents * 2, self.args.max_nodes)
            t.set_postfix(Agents=self.n_agents, Nodes=self.n_nodes)

            # print('Init...')
            self.initialize()
            # print('Exe...')
            self.exe()
            # print('Train...')
            self.train()
            # print('Eval...')
            self.evaluate()
            self.finalize()

    def exe(self):

        mems = self.exec_pool.map(self.gather_experience, range(self.args.num_processes))

        memory = []
        for mem in mems:
            memory += mem
        self.critic.memory.memory = memory

    def initialize(self):
        self.env = env(n_agent=self.n_agents, board_name='d_{}_{}_random'.format(self.n_nodes, self.n_nodes * 2))
        # print('Agents: {} Node: {}'.format(self.n_agents, self.n_nodes))
        # self.env = env(n_agent=n_agents, board_name='d_8node')
        # self.env = wrappers.PureGraphObservationWrapper(self.env)
        self.env = wrappers.GeometricObservationWrapper(self.env)
        # self.env = wrappers.GraphObservationWrapper(self.env)
        self.env = wrappers.SparseRewardWrapper(self.env)
        # self.env = wrappers.SparseRewardWrapper(self.env)
        self.env.reset()
        self.feat_size = 4
        self.type_size = self.n_nodes
        self.critic = DVNAgent(self.n_nodes, self.n_agents, self.feat_size, self.hidden_size, self.device)
        self.last_critic = DVNAgent(self.n_nodes, self.n_agents, self.feat_size, self.hidden_size, self.device)
        if self.episode == 0:
            torch.save(self.critic.state_dict(), self.save_path.format(self.episode))
            self.last_save_episode = self.episode

        self.critic.load_state_dict(torch.load(self.save_path.format(self.last_save_episode)))
        self.last_critic.load_state_dict(self.critic.state_dict())

    def gather_experience(self, _):
        if self.args.cuda:
            self.device = 'cuda:{}'.format(np.random.randint(7))
            self.critic.to(self.device)
        self.env.reset()
        id = np.random.randint(self.n_agents)
        for i, agent_id in enumerate(self.env.agent_iter(max_iter=self.args.num_steps)):
            state, _, _, info = self.env.last()

            self.critic.eval()

            if agent_id != id and False:
                task_id = state['task_id']
                action = SAMPLING[task_id](self.env.unwrapped.board, agent_id)
            else:
                action_scores, v = self.critic.predict(self.env.unwrapped.board, agent_id)
                deterministic, valid_actions = self.env.unwrapped.board.valid_actions(agent_id)
                action_id = np.random.choice(len(action_scores), p=action_scores)
                # action_id = np.argmax(action_scores)
                if len(action_scores) != len(valid_actions):
                    print(valid_actions, action_scores, action_id, self.env.unwrapped.board.state)
                action = valid_actions[action_id]

            last_data = state['data']
            self.env.step(action)
            state, _, _, info = self.env.last()
            data = state['data']
            reward = state['rewards']
            done = state['dones']

            if i == self.args.num_steps - 1:
                rew = [self.env.reward(i, True) for i in range(self.n_agents)]
                reward = rew
            data.reward = torch.tensor(reward)
            data.done = torch.tensor(all(done))
            data.task_id = state['task_id']
            data.last_x = last_data.x
            data.last_edge_index = last_data.edge_index
            data.last_edge_attr = last_data.edge_attr
            data.last_edge_type = last_data.edge_type
            data.last_node_type = last_data.node_type

            # make a transition and save to replay memory
            self.critic.save_memory(data)

            if all(done):
                # print('done')
                break

        return self.critic.memory.memory[:-1]

    def train(self):
        self.critic.train()
        # if self.critic.train_start():
        t = tqdm(range(self.args.train_epoch), desc='Train Epoch')
        for _ in t:
            loss = self.critic.train_()
            t.set_postfix(Loss=loss)
        # print('Trained')
        # return True
        # return False

    def eval(self, ids):
        j, id = ids
        self.env.reset()
        if self.args.cuda:
            self.device = 'cuda:{}'.format(np.random.randint(7))
            self.critic.to(self.device)
            self.last_critic.to(self.device)
        # id = np.random.randint(self.n_agents)
        for i, agent_id in enumerate(self.env.agent_iter(max_iter=self.args.num_steps)):
            if agent_id != id and j > self.args.eval_iter // 2 and False:
                state, _, _, _ = self.env.last()
                task_id = state['task_id']
                action = SAMPLING[task_id](self.env.unwrapped.board, agent_id)
            else:
                model = self.critic if agent_id == id else self.last_critic
                # Use Model to Gather Future State per Valid Actions
                deterministic, valid_actions = self.env.unwrapped.board.valid_actions(agent_id)
                action_scores, v = model.predict(self.env.unwrapped.board, agent_id)

                action_id = np.random.choice(len(action_scores), p=action_scores)
                # action_id = np.argmax(action_scores)
                action = valid_actions[action_id]

            self.env.step(action)
            state, _, _, _ = self.env.last()
            if all(state['dones']):
                return state['rewards'][id]
                # # print(id, state['rewards'])
                # if j > self.args.eval_iter // 2:
                #     y += state['rewards'][id]
                # else:
                #     w += state['rewards'][id]
                # break
        return 0

    def evaluate(self):
        self.critic.eval()
        self.last_critic.eval()
        w = 0
        y = 0
        ids = [i for i in range(self.n_agents)] * self.args.eval_iter
        t = self.eval_pool.map(self.eval, enumerate(ids))
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

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # torch.set_num_threads(1)
    i = 7
    device = torch.device("cuda:{}".format(i) if args.cuda else "cpu")

    coach = Coach(args, device)

    coach.execute()


if __name__ == "__main__":
    main()
