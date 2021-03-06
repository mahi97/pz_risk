import math
import random
import numpy as np

from core.gamestate import GameState


def sample_reinforce(board, player):
    nodes = board.player_nodes(player)
    return np.random.choice(nodes)  # [(0 if n not in nodes else r2[n]) for n in board.g.nodes()]


def sample_card(board, player):
    return np.random.randint(2) if len(board.players[player].cards) < 5 else 1


def sample_attack(board, player):
    edges = board.player_attack_edges(player)
    index = np.random.randint(len(edges))
    attack_finished = np.random.randint(10) > 7
    return attack_finished, edges[index]


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
    skip = np.random.randint(10) > 7
    return skip, src, trg, num_unit


SAMPLING = {
    GameState.Reinforce: sample_reinforce,
    GameState.Attack: sample_attack,
    GameState.Fortify: sample_fortify,
    GameState.Card: sample_card,
    GameState.Move: sample_move,
    GameState.EndTurn: lambda b, p: None
}
