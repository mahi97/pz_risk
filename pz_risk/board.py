import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
import random
from enum import Enum
from collections import Iterable

from attack_utils import single_roll

BOARDS = {}


def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item


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
        self.last_attack = (None, None)

    def player_nodes(self, player):
        return [n[0] for n in self.g.nodes(data=True) if n[1]['player'] == player]

    def cc(self, player, node, visited):
        gp = lambda x: self.g.nodes[x]['player']
        visited[node] = True

    def DFS(self, node, player, visited):
        gp = lambda x: self.g.nodes[x]['player']
        visited[node] = True
        for a in self.g.adj[node]:
            if gp(a) == player and visited[a] is False:
                self.DFS(a, player, visited)

    def player_connected_components(self, player):
        nodes = self.player_nodes(player)
        visited = {n: False for n in nodes}
        cc = []
        for n in nodes:
            if visited[n] is False:
                self.DFS(n, player, visited)
                cc.append([k for k, v in visited.items() if v and k not in flatten(cc)])
        return cc

    def player_attack_edges(self, player):
        ee = []
        for e in self.g.edges:
            if not self.g.nodes[e[0]]['player'] != self.g.nodes[e[1]]['player']:
                continue
            if player == self.g.nodes[e[0]]['player'] and self.g.nodes[e[0]]['units'] >= 2:
                ee.append(e)
            if player == self.g.nodes[e[1]]['player'] and self.g.nodes[e[1]]['units'] >= 2:
                ee.append((e[1], e[0]))

        if len(ee) == 0:
            print(ee)
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

    def step(self, agent, state, actions):

        if state == GameState.Reinforce:
            for i, action in enumerate(actions):
                current_unit = self.g.nodes[i + 1]['units']
                nx.set_node_attributes(self.g, {i + 1: current_unit + action}, 'units')
        elif state == GameState.Attack:
            src = actions[0]
            trg = actions[1]
            src_unit = self.g.nodes[src]['units']
            trg_unit = self.g.nodes[trg]['units']
            src_loss, trg_loss = single_roll(src_unit - 1, trg_unit)
            nx.set_node_attributes(self.g, {src: src_unit - src_loss}, 'units')
            nx.set_node_attributes(self.g, {trg: trg_unit - trg_loss}, 'units')

            if self.g.nodes[trg]['units'] == 0:
                self.g.nodes[trg]['player'] = self.g.nodes[src]['player']
                self.g.nodes[trg]['units'] = self.g.nodes[src]['units']
                self.g.nodes[src]['units'] = 1
                self.last_attack = (src, trg)
                return True
        elif state == GameState.Move:
            self.g.nodes[self.last_attack[0]]['units'] += actions
            self.g.nodes[self.last_attack[1]]['units'] -= actions
        elif state == GameState.Fortify:
            src_unit = self.g.nodes[actions[0]]['units']
            trg_unit = self.g.nodes[actions[1]]['units']
            nx.set_node_attributes(self.g, {actions[0]: src_unit - actions[2]}, 'units')
            nx.set_node_attributes(self.g, {actions[1]: trg_unit + actions[2]}, 'units')
        return False


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
