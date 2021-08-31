import math
import random
import numpy as np

from pz_risk.core.gamestate import GameState


def sample_reinforce(board, player):
    num_units = board.players[player].placement
    nodes = board.player_nodes(player)
    branches = math.comb(num_units + len(nodes) - 1, num_units)
    index = np.random.randint(branches)
    r = sorted([(index // num_units ** i) % num_units for i in range(len(nodes))])
    r.append(num_units)
    r2 = {n: r[i + 1] - r[i] for i, n in zip(range(len(nodes)), random.sample(nodes, len(nodes)))}
    return [(0 if n not in nodes else r2[n]) for n in board.g.nodes()]


def sample_card(board, player):
    return np.random.randint(2) if len(board.players[player].cards) < 5 else 1


def sample_attack(board, player):
    edges = board.player_attack_edges(player)
    index = np.random.randint(len(edges))
    return edges[index]


def sample_move(board, player):
    u = board.g.nodes[board.last_attack[1]]['units'] - 3
    return np.random.randint(u) if u > 0 else 0


def sample_fortify(board, player):
    cc = board.player_connected_components(player)
    branches = []
    potential_sources = []
    for c in cc:
        potential_source = [node for node in c if board.g.nodes[node]['units'] >= 2]
        branches.append(len(potential_source) * (len(c) - 1))
        potential_sources.append(potential_source)

    index = np.random.randint(sum(branches))
    src = -1
    trg = -1
    for b, c, ps in zip(branches, cc, potential_sources):
        if index < b:
            src = ps[index // len(c) - 1]
            trg = c[index % len(c)] if index % len(c) < index // len(c) else c[(index + 1) % len(c)]
            break
        index -= b
    num_unit = np.random.randint(board.g.nodes[src]['units'] - 1)
    return src, trg, num_unit


SAMPLING = {
    GameState.Reinforce: sample_reinforce,
    GameState.Attack: sample_attack,
    GameState.Fortify: sample_fortify,
    GameState.Card: sample_card,
    GameState.Move: sample_move
}
