import os
import torch
import random
import numpy as np

from risk_env import env
import training.utils as utils
from training.dvn2 import DVNAgent
from training.arguments import get_args
import wrappers

from agents.sampling import SAMPLING
from agents.value import get_future, get_attack_dist
from utils import get_feat_adj_from_board, get_feat_adj_type_from_board
from agents import GreedyAgent, RandomAgent
from copy import deepcopy

from tqdm import tqdm
import matplotlib.pyplot as plt


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    n_agents = 2
    e = env(n_agent=n_agents, board_name='d_8node')
    e = wrappers.PureGraphObservationWrapper(e)
    # e = wrappers.GraphObservationWrapper(e)
    e = wrappers.SparseRewardWrapper(e)
    # e = wrappers.SparseRewardWrapper(e)
    e.reset()
    _, _, _, info = e.last()
    n_nodes = info['nodes']
    n_agents = info['agents']
    max_episode = 3000
    max_epi_step = 300

    epsilon = 1.0
    epsilon_min = 0.0005
    decay_rate = 0.05

    feat_size = e.observation_spaces['feat'].shape[0]
    hidden_size = 20
    type_size = 2
    critic = DVNAgent(n_nodes, n_agents, type_size, feat_size, hidden_size, device)
    save_path = './mini_0/'
    load = 0
    critic.load_state_dict(torch.load(os.path.join(save_path, str(load) + ".pt")))
    loss_list = []
    reward_list = []

    # players = [None]
    # players = [RandomAgent(i) for i in range(1, 6)]

    for episode in tqdm(range(load, max_episode+1)):

        e.reset()
        state, _, _, _ = e.last()
        loss_epi = []
        reward_epi = []
        id = np.random.randint(n_agents)
        for i, agent_id in enumerate(e.agent_iter(max_iter=max_epi_step)):
            state, _, _, info = e.last()

            critic.eval()
            if agent_id != id:
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
                        action_scores.append(10000)
                    else:
                        action_scores.append(critic(sim_feat, sim_adj, sim_type).detach().cpu().numpy()[:, n_nodes + agent_id])
                # if state['task_id'].value == 3:
                #     print(valid_actions)
                #     print(action_scores)
                action = valid_actions[np.argmax(action_scores)]

            before_feat = torch.tensor(state['feat'], dtype=torch.float32, device=device).reshape(-1,
                                                                                                  n_nodes + n_agents,
                                                                                                  feat_size)
            before_adj = torch.tensor(state['adj'], dtype=torch.float32, device=device).reshape(-1, n_nodes + n_agents,
                                                                                                n_nodes + n_agents)

            e.step(action)
            state, _, _, info = e.last()
            feat = torch.tensor(state['feat'], dtype=torch.float32, device=device).reshape(-1, n_nodes + n_agents,
                                                                                           feat_size)
            adj = torch.tensor(state['adj'], dtype=torch.float32, device=device).reshape(-1, n_nodes + n_agents,
                                                                                         n_nodes + n_agents)
            types = torch.tensor(state['type'], dtype=torch.float32, device=device).reshape(-1, n_nodes + n_agents)
            reward = torch.tensor(state['rewards'], dtype=torch.float32, device=device).reshape(-1, n_agents)
            done = torch.tensor(state['dones'], dtype=torch.bool, device=device).reshape(-1, n_agents)
            reward_epi.append(reward.cpu().numpy()[0])
            # e.render()
            if all(state['dones']):
                break

            if i == max_epi_step - 1:
                rew = [e.reward(i, True) for i in range(n_agents)]
                # rew2 = [0 for i in range(n_agents)]
                reward = torch.tensor([rew], dtype=torch.float32, device=device)
                # print('wtf')
            # make a transition and save to replay memory
            # print(reward)
            transition = [before_feat, before_adj, reward, feat, adj, types, done]
            critic.save_memory(transition)
            critic.train()
            if critic.train_start():
                loss = critic.train_()
                loss_epi.append(loss)
                # print('Loss: {}, Reward: {}'.format(loss, reward))

        if i < max_epi_step - 1:
            print('hey')
        # print(epsilon)
        if epsilon > epsilon_min:
            epsilon -= decay_rate
        else:
            epsilon = epsilon_min

        if critic.train_start():
            loss_list.append(sum(loss_epi) / len(loss_epi))
        # plt.show()
        e.close()
        reward_list.append(sum(reward_epi))

        if episode % 10 == 0 and episode // 10 > 0:
            print(episode + 1, reward_list[-1], loss_list[-1])
            torch.save(critic.state_dict(), os.path.join(save_path, str(episode // 10) + ".pt"))

    return loss_list, reward_list


if __name__ == "__main__":
    main()
