import os

import networkx as nx
import json
from enum import Enum

from pz_risk.core.card import Card, CardType, CARD_FIX_SCORE
from pz_risk.core.player import Player

from pz_risk.utils import *
from pz_risk.core.gamestate import GameState

BOARDS = {}


class Board:
    def __init__(self, graph: nx.Graph, info, pos=None):
        self.g = graph
        self.pos = pos
        self.players = []
        self.cards = []
        self.last_attack = (None, None)
        self.state = GameState.StartTurn
        self.info = info

    def can_fortify(self, player):
        cc = self.player_connected_components(player)
        branches = []
        potential_sources = []
        for c in cc:
            potential_source = [node for node in c if self.g.nodes[node]['units'] >= 2]
            branches.append(len(potential_source) * (len(c) - 1))
            potential_sources.append(potential_source)
        return sum(branches) > 0

    def can_attack(self, player):
        return len(self.player_attack_edges(player))

    def can_card(self, player):
        if self.players[player].num_cards() >= 5:
            return True

        if self.players[player].num_cards() >= 3 and self.state == GameState.StartTurn:
            ct = self.players[player].cards
            if max(ct.values()) + len(ct[CardType.Wild]) >= 3 or min(ct.values()) == 1:
                return True
        return False

    def calc_units(self, player):
        nodes_controlled = len([i for i in self.g.nodes(data='player') if i[1] == player])
        node_unit = max(3, nodes_controlled // 3)

        group = {gid+1: True for gid in range(self.info['num_of_group'])}
        for n in self.g.nodes(data=True):
            if n[1]['player'] != player:
                group[n[1]['gid']] = False
        group_unit = sum([self.info['group_reward'][str(k)] for k, v in group.items() if v])

        return group_unit + node_unit


    def next_state(self, player, state, attack_succeed, attack_finished, game_over):
        next_state = {
            GameState.StartTurn: GameState.Card if self.can_card(player) else GameState.Reinforce,
            GameState.Card: GameState.Reinforce,
            GameState.Reinforce:
                GameState.Attack if self.can_attack(player)
                else GameState.Fortify if self.can_fortify(player)
                else GameState.StartTurn,
            GameState.Attack:
                GameState.EndTurn if game_over
                else GameState.Move if attack_succeed
                else GameState.Fortify if attack_finished and self.can_fortify(player)
                else GameState.StartTurn if attack_finished
                else GameState.Attack if self.can_attack(player)
                else GameState.Fortify if self.can_fortify(player)
                else GameState.StartTurn,
            GameState.Move:
                GameState.Card if self.can_card(player)
                else GameState.Attack if self.can_attack(player)
                else GameState.Fortify if self.can_fortify(player)
                else GameState.StartTurn,
            GameState.Fortify: GameState.StartTurn,
            GameState.EndTurn: GameState.EndTurn
        }
        self.state = next_state[state]

    def player_nodes(self, player):
        return [n[0] for n in self.g.nodes(data=True) if n[1]['player'] == player]

    # def cc(self, player, node, visited):
    #     gp = lambda x: self.g.nodes[x]['player']
    #     visited[node] = True

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
        return ee

    def reset(self, n_agent, n_unit_per_agent, n_cell_per_agent):
        n_cells = self.g.number_of_nodes()
        assert n_cell_per_agent * n_agent == n_cells

        remaining_cells = [i for i in self.g.nodes()]
        for i in range(n_agent):
            cells = random.sample(remaining_cells, n_cell_per_agent)
            remaining_cells = [c for c in remaining_cells if c not in cells]
            nx.set_node_attributes(self.g, {c: int(1) for c in cells}, 'units')
            nx.set_node_attributes(self.g, {c: i for c in cells}, 'player')
            t = n_unit_per_agent - n_cell_per_agent
            left_unit = t
            for j, cell in enumerate(cells):
                if left_unit <= 0:
                    break
                x = left_unit if j + 1 == len(cells) else random.randint(1, min(left_unit, t // 3))
                left_unit -= x
                nx.set_node_attributes(self.g, {cell: int(x + 1)}, 'units')
        self.state = GameState.Reinforce

        # Initial Cards
        self.cards = [Card(i, CardType(np.random.randint(len(CardType) - 1))) for i in self.g.nodes()]
        self.cards += [Card(-1, CardType.Wild) for _ in range(self.info['num_of_wild'])]

        # Initial Player
        self.players = [Player(i, 3) for i in range(n_agent)]

    def apply_best_match(self, player):
        ct = {t: [] for t in [CardType.Infantry, CardType.Cavalry, CardType.Artillery, CardType.Wild]}
        for card in self.players[player].cards:
            if card.type == CardType.Wild:
                ct[CardType.Infantry] += 1
                ct[CardType.Cavalry] += 1
                ct[CardType.Artillery] += 1
            ct[card.type] += 1
        if min(ct.values()) == 1:
            match_type = CardType.Wild
        else:
            match_type = CardType.Artillery if ct[CardType.Artillery] >= 3 \
                else CardType.Cavalry if ct[CardType.Cavalry] >= 3 else CardType.Infantry
        self.players[player].placement += CARD_FIX_SCORE[match_type]

    def give_card(self, player):
        self.players[player].deserve_card = False
        remaining_cards = [c for c in self.cards if c.owner == -1]
        card = np.random.choice(remaining_cards)
        self.players[player].cards[card.type].append(card)
        card.owner = player

    def step(self, agent, actions):
        attack_succeed = False

        if self.state == GameState.StartTurn:
            self.players[agent].placement = self.calc_units(agent)
            self.next_state(agent, self.state, False, False, False)
        elif self.state == GameState.Reinforce:
            for i, action in enumerate(actions):
                current_unit = self.g.nodes[i + 1]['units']
                nx.set_node_attributes(self.g, {i + 1: int(current_unit + action)}, 'units')
        elif self.state == GameState.Card:
            if actions:
                self.apply_best_match(agent)
        elif self.state == GameState.Attack:
            src = actions[0]
            trg = actions[1]
            src_unit = self.g.nodes[src]['units']
            trg_unit = self.g.nodes[trg]['units']
            src_loss, trg_loss = single_roll(src_unit - 1, trg_unit)
            nx.set_node_attributes(self.g, {src: int(src_unit - src_loss)}, 'units')
            nx.set_node_attributes(self.g, {trg: int(trg_unit - trg_loss)}, 'units')

            if self.g.nodes[trg]['units'] == 0:
                self.g.nodes[trg]['player'] = self.g.nodes[src]['player']
                self.g.nodes[trg]['units'] = self.g.nodes[src]['units'] - 1
                self.g.nodes[src]['units'] = 1
                self.last_attack = (src, trg)
                attack_succeed = True
                self.players[agent].deserve_card = True
        elif self.state == GameState.Move:
            self.g.nodes[self.last_attack[0]]['units'] += int(actions)
            self.g.nodes[self.last_attack[1]]['units'] -= int(actions)
        elif self.state == GameState.Fortify:
            self.g.nodes[actions[0]]['units'] -= int(actions[2])
            self.g.nodes[actions[1]]['units'] += int(actions[2])

        self.next_state(agent, self.state, attack_succeed, False, len(self.player_nodes(agent)) == len(self.g.nodes()))
        if self.state == GameState.StartTurn and self.players[agent].deserve_card:
            self.give_card(agent)

def register_map(name, filepath):
    global BOARDS
    f = open(filepath)
    m = json.load(f)
    g = nx.Graph()
    g.add_nodes_from([(cell['id'], cell) for cell in m['cells']])
    g.add_edges_from([e for e in m['edges']])
    assert min([d[1] for d in g.degree()]) > 0

    BOARDS[name] = Board(g, m['info'])


register_map('world', './maps/world.json')
