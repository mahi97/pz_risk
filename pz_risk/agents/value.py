import math

import numpy as np
from core.gamestate import GameState

from copy import deepcopy

# From https://web.stanford.edu/~guertin/risk.notes.html
win_rate = np.array([
    [[0.417, 0.583, 0.],        # 1 vs 1
     [0.255, 0.745, 0.]],       # 1 vs 2
    [[0.579, 0.421, 0.],      # 2 vs 1
     [0.228, 0.324, 0.448]],    # 2 vs 2
    [[0.660, 0.340, 0.],      # 3 vs 1
     [0.371, 0.336, 0.293]]   # 3 vs 2
])

d3 = {}


def warm_up():
    for i in range(1, 100):
        for j in range(1, 100):
            for r in range(-j, i):
                get_chance(i, j, r)


def get_chance(attack_unit, defense_unit, left):
    global win_rate, d3
    i_a = min(attack_unit - 1, 2)
    i_d = min(defense_unit - 1, 1)
    if (attack_unit, defense_unit, left) in d3:
        c = d3[(attack_unit, defense_unit, left)]
        return c

    c = 0.0
    if left < -defense_unit or left > attack_unit:
        c = 0.0
    elif defense_unit < 0 or attack_unit < 0:
        c = 0.0
    elif attack_unit == 0:
        if left == -defense_unit:
            c = 1.0
        else:
            c = 0.0
    elif defense_unit == 0:
        if left == attack_unit:
            c = 1.0
        else:
            c = 0.0
    else:
        c = win_rate[i_a, i_d, 0] * get_chance(attack_unit, defense_unit - min(min(i_a, i_d) + 1, 2), left) + \
            win_rate[i_a, i_d, 1] * get_chance(attack_unit - 1, defense_unit - min(i_a, 1), left) + \
            win_rate[i_a, i_d, 2] * get_chance(attack_unit - 2, defense_unit, left)
    d3[(attack_unit, defense_unit, left)] = c
    return c


def get_future(dist, mode='safe', risk=0.0):
    if mode == 'safe':
        return dist[0][0]
    elif mode == 'risk':
        sum_risk = 0.0
        for d in dist:
            sum_risk += d[1]
            if sum_risk > risk:
                return d[0]
    elif mode == 'most':
        max_index = np.argmax([d[1] for d in dist])
        return dist[max_index][0]
    elif mode == 'two':
        left = [d[1] for d in dist if d[1] < 0]
        right = [d[1] for d in dist if d[1] > 0]
        left_score = dist[np.argmax(left)][0] * sum(left)
        right_score = dist[np.argmax(right)][0] * sum(right)
        return left_score + right_score
    elif mode == 'all':
        final = sum([d[0] * d[1] for d in dist])
        return final


def manual_value(board, player):
    num_lands = len(board.player_nodes(player)) * 5
    num_units = board.player_units(player)
    group_reward = board.player_group_reward(player) * 10
    num_cards = sum([len(c) for c in board.players[player].cards.values()])
    return num_lands + num_units + group_reward + num_cards


def man_q_deterministic(board, player, action):
    sim = deepcopy(board)
    sim.step(player, action)
    return manual_value(sim, player)


def man_q_attack(board, player, action):
    attack_finished = action[0]
    sim = deepcopy(board)
    if not attack_finished:
        src = action[1][0]
        trg = action[1][1]
        src_unit = board.g.nodes[src]['units']
        trg_unit = board.g.nodes[trg]['units']
        dist = [(i, get_chance(src_unit, trg_unit, i)) for i in range(-trg_unit, src_unit+1)]
        left = get_future(dist, mode='most')
        sim.step(player, action, left)
    return manual_value(sim, player)


def manual_q(state, player, action):
    Q = {
        GameState.Reinforce: man_q_deterministic,
        GameState.Attack: man_q_attack,
        GameState.Fortify: man_q_deterministic,
        GameState.Card: man_q_deterministic,
        GameState.Move: man_q_deterministic,
        GameState.EndTurn: lambda b, p: None
    }

    return Q[state['game_state']](state['board'], player, action)


def manual_advantage(state, player, action):
    return manual_q(state, player, action) - manual_value(state['board'], player)
