import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
import random
from enum import Enum

BOARDS = {}


class GameState(Enum):
    Start = 0
    Card = 1
    Reinforce = 2
    Attack = 3
    Move = 4
    Fortify = 5
    End = 6


class CardType(Enum):
    Null = 0
    Infantry = 1
    Cavalry = 2
    Artillery = 3
    Wild = 4


class Card:
    def __init__(self):
        self.type = CardType.Null
        self.effective_node = -1


class Player:
    def __init__(self):
        self.cards = []
        self.num_of_unit_to_place = 0


class Board:
    def __init__(self, graph: nx.Graph, pos=None):
        self.g = graph
        self.pos = pos
        self.player = []

    def player_nodes(self, player):
        return [n[0] for n in self.g.nodes(data=True) if n[1]['player'] == player]

    def player_attack_edges(self, player):
        ee = []
        for e in self.g.edges:
            a = self.g.nodes[e[0]]['player'] != self.board.g.nodes[e[1]]['player']
            b = player == self.g.nodes[e[0]]['player'] and self.g.nodes[e[0]]['units'] >= 2
            c = player == self.g.nodes[e[1]]['player'] and self.g.nodes[e[1]]['units'] >= 2
            if a and (b or c):
                ee.append(e)
        return ee

    def reset(self, n_agent, n_unit_per_agent, n_cell_per_agent):
        self.player = [Player() for i in range(n_agent)]
        n_cells = self.g.number_of_nodes()
        assert n_cell_per_agent * n_agent == n_cells
        remaining_cells = [i for i in self.g.nodes()]
        for i in range(n_agent):
            cells = random.sample(remaining_cells, n_cell_per_agent)
            remaining_cells = [c for c in remaining_cells if c not in cells]
            nx.set_node_attributes(self.g, {c: 1 for c in cells}, 'units')
            nx.set_node_attributes(self.g, {c: i for c in cells}, 'player')
            t = n_unit_per_agent - n_cell_per_agent
            left_unit = t
            for j, cell in enumerate(cells):
                if left_unit <= 0:
                    break
                x = left_unit if j + 1 == len(cells) else random.randint(1, min(left_unit, t // 3))
                print(x)
                left_unit -= x
                nx.set_node_attributes(self.g, {cell: x + 1}, 'units')

    def step(self, agent, state, action):
        if state == GameState.Reinforce:  # Reinforce
            self.add_unit(action[0], action[1])
        elif state == GameState.Attack:  # Attack
            self.attack(action[0], action[1])
        elif state == GameState.Fortify:  # Fortify
            self.fortify(action[0], action[1], action[2])


def register_map(name, filepath):
    global BOARDS
    f = open(filepath)
    m = json.load(f)
    g = nx.Graph()
    g.add_nodes_from([(cell['id'], cell) for cell in m['cells']])
    g.add_edges_from([e for e in m['edges']])
    assert min([d[1] for d in g.degree()]) > 0
    BOARDS[name] = Board(g, m['pos'])


register_map('world', 'maps/world.json')
