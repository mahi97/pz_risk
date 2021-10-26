# https://github.com/rmcarthur/gym-risk/blob/master/gym_risk/attack_utils.py

from loguru import logger
import random
import numpy as np
from collections import Iterable

from copy import deepcopy
import networkx as nx

rng = np.random.default_rng()
sided_die = 6
attack_max = 3
defend_max = 2


def single_roll(attack: int, defend: int) -> (int, int):
    attack_roll = np.sort(rng.integers(1, sided_die + 1, min(attack_max, attack)))[::-1]
    defend_roll = np.sort(rng.integers(1, sided_die + 1, min(defend_max, defend)))[::-1]
    # logger.debug(f"Attack roll: {attack_roll}")
    # logger.debug(f"defend roll: {defend_roll}")
    max_loss = min(len(attack_roll), len(defend_roll))
    attack_wins = np.sum([i > j for i, j in zip(attack_roll, defend_roll)])
    attack_loss = max_loss - attack_wins
    defend_loss = attack_wins
    return attack_loss, defend_loss


def to_one_hot(num, maximum):
    return [1 if i == num else 0 for i in range(maximum)]


def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item


def get_feat_adj_from_board(board, player, n_agents, n_grps):
    feats = []
    for node in board.g.nodes(data=True):
        feats.append([
            0,
            node[1]['units'],
            node[1]['gid'],
            board.info['group_reward'][str(node[1]['gid'])],
            -1, -1  # Player
        ])
    # feat_player = []
    for p in range(n_agents):
        feats.append([
            -1, 0, -1, 0,
            1 if board.can_card(p) else 0,
            1 if p == player else 0
        ])
    temp_g = nx.Graph(deepcopy(board.g))
    temp_g.add_nodes_from(['p' + str(i) for i in range(n_agents)])
    edges = [[node[0], 'p' + str(node[1]['player'])] for node in temp_g.nodes(data=True) if 'player' in node[1]]

    # edges = []
    # for i in range(n_agents):
    #     edges += [[node[0], 'p' + str(i)] for node in temp_g.nodes(data=True) if 'player' in node[1]]
    temp_g.add_edges_from(edges)

    feats = [list(flatten([to_one_hot(feat[0], 2), feat[1], to_one_hot(feat[2], n_grps), feat[3],
                           to_one_hot(feat[4], 2), to_one_hot(feat[5], 2)])) for feat in feats]
    adj = nx.adjacency_matrix(temp_g).todense() + np.eye(len(temp_g.nodes()))
    return feats, adj
