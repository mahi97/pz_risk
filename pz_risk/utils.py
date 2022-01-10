# https://github.com/rmcarthur/gym-risk/blob/master/gym_risk/attack_utils.py

from loguru import logger
import random
import numpy as np
from collections import Iterable

from copy import deepcopy
import networkx as nx

import torch
from torch_geometric.data import HeteroData, Data
import torch_geometric.transforms as T

import matplotlib.pyplot as plt

rng = np.random.default_rng()
sided_die = 6
attack_max = 3
defend_max = 2
COLORS = [
    'tab:red',
    'tab:blue',
    'tab:green',
    'tab:purple',
    'tab:pink',
    'tab:cyan',
    'tab:red',
    'tab:blue',
    'tab:green',
    'tab:purple',
    'tab:pink',
    'tab:cyan',
    'tab:red',
    'tab:blue',
    'tab:green',
    'tab:purple',
    'tab:pink',
    'tab:cyan'
]


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


def draw_pure(G: nx.Graph, n_agents):
    pos = nx.kamada_kawai_layout(G)
    options = {"edgecolors": "tab:gray", "node_size": 800, "alpha": 0.9}
    for agent in range(n_agents):
        nx.draw_networkx_nodes(G, pos,
                               nodelist=[c[0] for c in G.nodes(data=True) if
                                         'player' in c[1] and c[1]['player'] == agent],
                               node_color=COLORS[agent], **options)
    # nx.draw_networkx_nodes(G, pos, nodelist=[4, 5, 6, 7], node_color="tab:blue", **options)

    # edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

    # some math labels
    # labels = {c[0]: c[1]['units'] if 'units' in c[1] else c[1]['label'] for c in G.nodes(data=True)}
    # nx.draw_networkx_labels(G, pos, labels, font_size=22, font_color="black")
    plt.tight_layout()
    plt.axis("off")


def draw(graph: nx.MultiDiGraph):
    # pos = {}
    pos = nx.kamada_kawai_layout(graph)
    # pos = nx.random_layout(graph)
    cnts = {}
    for node in graph.nodes(data='type'):
        if node[1] not in cnts:
            cnts[node[1]] = -1
        cnts[node[1]] += 1
    for node in graph.nodes(data=True):
        if node[1]['type'] == 'player':
            pos[node[0]] = -1 + (node[1]['label'] / cnts[node[1]['type']] * 2), 0.5
        # if node[1]['type'] == 'country':
        #     pos[node[0]] = -1+(node[1]['id']/cnts[node[1]['type']]*2), np.random.rand()/2.3 - 0.4
        if node[1]['type'] == 'unit':
            pos[node[0]] = -1 + (node[1]['label'] / cnts[node[1]['type']] * 2), -0.5
    options = {"edgecolors": "tab:gray", "node_size": 800, "alpha": 0.9}
    nx.draw_networkx_nodes(graph, pos, nodelist=[c[0] for c in graph.nodes(data='type') if c[1] == 'player'],
                           node_color=COLORS[0], **options)
    nx.draw_networkx_nodes(graph, pos, nodelist=[c[0] for c in graph.nodes(data='type') if c[1] == 'country'],
                           node_color=COLORS[1], **options)
    nx.draw_networkx_nodes(graph, pos, nodelist=[c[0] for c in graph.nodes(data='type') if c[1] == 'unit'],
                           node_color=COLORS[2], **options)
    nx.draw_networkx_nodes(graph, pos, nodelist=[c[0] for c in graph.nodes(data='type') if c[1] == 'dummy'],
                           node_color=COLORS[3], **options)
    # nx.draw_networkx_nodes(G, pos, nodelist=[4, 5, 6, 7], node_color="tab:blue", **options)

    nx.draw_networkx_edges(graph, pos,
                           edgelist=[(c[0], c[1]) for c in graph.edges(data='type') if c[2] and c[2] == 'attack'],
                           width=1.0, alpha=0.5, connectionstyle='arc3, rad = 0.1', edge_color=COLORS[0])
    nx.draw_networkx_edges(graph, pos,
                           edgelist=[(c[0], c[1]) for c in graph.edges(data='type') if c[2] and c[2] == 'own'],
                           width=1.0, alpha=0.5, connectionstyle='arc3, rad = 0.1', edge_color=COLORS[1])
    nx.draw_networkx_edges(graph, pos,
                           edgelist=[(c[0], c[1]) for c in graph.edges(data='type') if c[2] and c[2] == 'have'],
                           width=1.0, alpha=0.5, connectionstyle='arc3, rad = 0.1', edge_color=COLORS[2])
    nx.draw_networkx_edges(graph, pos,
                           edgelist=[(c[0], c[1]) for c in graph.edges(data='type') if c[2] and c[2] == 'fortify'],
                           width=1.0, alpha=0.5, connectionstyle='arc3, rad = 0.1', edge_color=COLORS[3])
    nx.draw_networkx_edges(graph, pos,
                           edgelist=[(c[0], c[1]) for c in graph.edges(data='type') if c[2] and c[2] == 'dummy_edge'],
                           width=1.0, alpha=0.5, connectionstyle='arc3, rad = 0.1', edge_color=COLORS[4])

    # nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5)

    # some math labels
    labels = {c[0]: c[1] for c in graph.nodes(data='label')}
    nx.draw_networkx_labels(graph, pos, labels, font_size=20, font_color="black")
    plt.tight_layout()
    plt.axis("off")
    plt.show()


def draw2(graph: nx.MultiDiGraph):
    # pos = {}
    pos = {k: (v[0] * 2 - 0.25, v[1] / 1.5) for k, v in nx.kamada_kawai_layout(graph).items()}
    # pos = nx.random_layout(graph)
    cnts = {}
    for node in graph.nodes(data='type'):
        if node[1] not in cnts:
            cnts[node[1]] = -1
        cnts[node[1]] += 1
    for node in graph.nodes(data=True):
        if node[1]['type'] == 'player':
            pos[node[0]] = -1 + (node[1]['label'] / cnts[node[1]['type']] * 2), 0.5
        # if node[1]['type'] == 'country':
        #     pos[node[0]] = -1+(node[1]['id']/cnts[node[1]['type']]*2), np.random.rand()/2.3 - 0.4
        if node[1]['type'] == 'unit':
            pos[node[0]] = -1 + (node[1]['label'] / cnts[node[1]['type']] * 2), -0.5
    options = {"edgecolors": "tab:gray", "node_size": 400, "alpha": 0.9}
    nx.draw_networkx_nodes(graph, pos, nodelist=[c[0] for c in graph.nodes(data='type') if c[1] == 'player'],
                           node_color=COLORS[0], **options)
    nx.draw_networkx_nodes(graph, pos, nodelist=[c[0] for c in graph.nodes(data='type') if c[1] == 'country'],
                           node_color=COLORS[1], **options)
    nx.draw_networkx_nodes(graph, pos, nodelist=[c[0] for c in graph.nodes(data='type') if c[1] == 'unit'],
                           node_color=COLORS[2], **options)
    # nx.draw_networkx_nodes(graph, pos, nodelist=[c[0] for c in graph.nodes(data='type') if c[1] == 'dummy'],
    #                        node_color=COLORS[3], **options)

    nx.draw_networkx_edges(graph, pos,
                           edgelist=[(c[0], c[1]) for c in graph.edges(data='type') if c[2] and c[2] == 'attack'],
                           width=2.0, alpha=0.5, connectionstyle='arc3, rad = 0.1', edge_color=COLORS[0])
    nx.draw_networkx_edges(graph, pos,
                           edgelist=[(c[0], c[1]) for c in graph.edges(data='type') if c[2] and c[2] == 'own'],
                           width=2.0, alpha=0.5, connectionstyle='arc3, rad = 0.1', edge_color=COLORS[1])
    nx.draw_networkx_edges(graph, pos,
                           edgelist=[(c[0], c[1]) for c in graph.edges(data='type') if c[2] and c[2] == 'have'],
                           width=2.0, alpha=0.5, connectionstyle='arc3, rad = 0.1', edge_color=COLORS[2])
    nx.draw_networkx_edges(graph, pos,
                           edgelist=[(c[0], c[1]) for c in graph.edges(data='type') if c[2] and c[2] == 'fortify'],
                           width=2.0, alpha=0.5, connectionstyle='arc3, rad = 0.1', edge_color=COLORS[3])
    # nx.draw_networkx_edges(graph, pos,
    #                        edgelist=[(c[0], c[1]) for c in graph.edges(data='type') if c[2] and c[2] == 'dummy_edge'],
    #                        width=1.0, alpha=0.5, connectionstyle='arc3, rad = 0.1', edge_color=COLORS[4])

    # nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5)

    # some math labels
    # labels = {c[0]: c[1] for c in graph.nodes(data='label')}
    # nx.draw_networkx_labels(graph, pos, labels, font_size=20, font_color="black")
    plt.tight_layout()
    plt.axis("off")
    plt.show()

def draw3(graph: nx.MultiDiGraph):
    # pos = {}
    pos = {k: (v[0] * 2 - 0.25, v[1] / 1.5) for k, v in nx.kamada_kawai_layout(graph).items()}
    # pos = nx.random_layout(graph)
    cnts = {}
    for node in graph.nodes(data='type'):
        if node[1] not in cnts:
            cnts[node[1]] = -1
        cnts[node[1]] += 1
    for node in graph.nodes(data=True):
        if node[1]['type'] == 'player':
            pos[node[0]] = -1 + (node[1]['label'] / cnts[node[1]['type']] * 2), 0.5
        # if node[1]['type'] == 'country':
        #     pos[node[0]] = -1+(node[1]['id']/cnts[node[1]['type']]*2), np.random.rand()/2.3 - 0.4
        if node[1]['type'] == 'unit':
            pos[node[0]] = -1 + (node[1]['label'] / cnts[node[1]['type']] * 2), -0.5
    options = {"edgecolors": "tab:gray", "node_size": 800, "alpha": 0.9}
    nx.draw_networkx_nodes(graph, pos, nodelist=[c[0] for c in graph.nodes(data='type') if c[1] == 'player'],
                           node_color=COLORS[0], **options)
    nx.draw_networkx_nodes(graph, pos, nodelist=[c[0] for c in graph.nodes(data='type') if c[1] == 'country'],
                           node_color=COLORS[1], **options)
    nx.draw_networkx_nodes(graph, pos, nodelist=[c[0] for c in graph.nodes(data='type') if c[1] == 'unit'],
                           node_color=COLORS[2], **options)
    # nx.draw_networkx_nodes(graph, pos, nodelist=[c[0] for c in graph.nodes(data='type') if c[1] == 'dummy'],
    #                        node_color=COLORS[3], **options)

    nx.draw_networkx_edges(graph, pos,
                           edgelist=[(c[0], c[1]) for c in graph.edges(data='type') if c[2] and c[2] == 'attack'],
                           width=1.0, edge_color=COLORS[0])
    nx.draw_networkx_edges(graph, pos,
                           edgelist=[(c[0], c[1]) for c in graph.edges(data='type') if c[2] and c[2] == 'own'],
                           width=1.0, edge_color=COLORS[1])
    nx.draw_networkx_edges(graph, pos,
                           edgelist=[(c[0], c[1]) for c in graph.edges(data='type') if c[2] and c[2] == 'have'],
                           width=1.0, edge_color=COLORS[2])
    nx.draw_networkx_edges(graph, pos,
                           edgelist=[(c[0], c[1]) for c in graph.edges(data='type') if c[2] and c[2] == 'fortify'],
                           width=1.0, edge_color=COLORS[3])
    # nx.draw_networkx_edges(graph, pos,
    #                        edgelist=[(c[0], c[1]) for c in graph.edges(data='type') if c[2] and c[2] == 'dummy_edge'],
    #                        width=1.0, alpha=0.5, connectionstyle='arc3, rad = 0.1', edge_color=COLORS[4])

    # nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5)

    # some math labels
    # labels = {c[0]: c[1] for c in graph.nodes(data='label')}
    # nx.draw_networkx_labels(graph, pos, labels, font_size=20, font_color="black")
    plt.tight_layout()
    plt.axis("off")
    plt.show()


def get_geom_from_board_single_arg(args):
    return get_geom_from_board(args[0], args[1], args[2])


def get_geom_from_board(board, n_agents, agent_id, normalize=True, self_loop=False, undirected=False):
    tmp_g = nx.MultiDiGraph(board.g)
    to_remove = [edge for edge in tmp_g.edges()]
    tmp_g.remove_edges_from(to_remove)
    all_units = sum([a[1] for a in board.g.nodes(data='units')])
    unit_norm = all_units if normalize else 1
    terr_norm = len(board.g.nodes) if normalize else 1
    for node in tmp_g.nodes(data=True):
        node[1]['type'] = 'country'
        node[1]['type_id'] = 0
        node[1]['id'] = node[0]
        node[1]['label'] = node[0]

    for agent in range(n_agents):
        tmp_g.add_edges_from(board.player_attack_edges(agent), type='attack', type_id=0)
        f_edges, f_weights = board.player_fortify_edges(agent)
        tmp_g.add_edges_from(f_edges, type='fortify', type_id=1)

        tmp_g.add_node('a_{}'.format(agent),
                       type='player', type_id=1,
                       label=agent, id=tmp_g.number_of_nodes())
        for node in tmp_g.nodes(data=True):
            if 'player' in node[1] and node[1]['player'] == agent:
                tmp_g.add_edge('a_{}'.format(agent), node[0], type='own', type_id=2)

    unit_nodes, unit_edges = [], []
    for node in tmp_g.nodes(data=True):
        if node[1]['type'] == 'country':
            unit_nodes += ['u_{}_{}'.format(node[0], u) for u in range(node[1]['units'])]
            unit_edges += [[node[0], 'u_{}_{}'.format(node[0], u)] for u in range(node[1]['units'])]
            pnode = 'a_{}'.format(node[1]['player'])
            unit_edges += [[pnode, 'u_{}_{}'.format(node[0], u)] for u in range(node[1]['units'])]

    # place_nodes, place_edges = [], []
    # for player in board.players:
    #     pnode = 'a_{}'.format(player.id)
    #     place_nodes += ['p_{}_{}'.format(pnode, u) for u in range(player.placement)]
    #     place_edges += [[pnode, 'p_{}_{}'.format(pnode, u)] for u in range(player.placement)]

    # order_edges = []
    # for player in board.players:
    #     pnode = 'a_{}'.format(player.id)
    #     qnode = 'a_{}'.format((player.id + 1) % len(board.players))
    #     order_edges.append([pnode, qnode])
    # tmp_g.add_edges_from(order_edges, type='dummy_order', type_id=4)


    # Add Dummy Node
    tmp_g.add_node('dummy', type='dummy', type_id=2,
                   label='D', id=tmp_g.number_of_nodes())
    for node in tmp_g.nodes():
        if node != 'dummy':
            tmp_g.add_edge('dummy', node, type='dummy_edge', type_id=3)
            tmp_g.add_edge(node, 'dummy', type='dummy_edge', type_id=3)

    # Add unit (will be removed later)
    tmp_g.add_nodes_from([(un,
                           {'type': 'unit', 'type_id': 3, 'label': i, 'id': tmp_g.number_of_nodes() + i}
                           ) for i, un in enumerate(unit_nodes)])
    tmp_g.add_edges_from(unit_edges, type='have', type_id=4)

    # tmp_g.add_nodes_from([(un,
    #                        {'type': 'placement', 'type_id': 4, 'label': i, 'id': tmp_g.number_of_nodes() + i}
    #                        ) for i, un in enumerate(place_nodes)])
    # tmp_g.add_edges_from(place_edges, type='place', type_id=5)

    # Generate Features:
    log = {
        'own': [],
        'have': [],
        'attack': [],
        'fortify': [],
        # 'place': [],
        }
    for node in tmp_g.nodes():
        cnts = {'own': 0, 'have': 0, 'attack': 0, 'fortify': 0}
        for nb in tmp_g.adj[node].items():
            edge_type = nb[1][0]['type']
            if edge_type.startswith('dummy_'):
                continue
            cnts[edge_type] += 1.0
        for k, v in cnts.items():
            log[k].append(v)
        # cnts['have'] /= unit_norm
        # cnts['own'] /= terr_norm
        # cnts['attack'] /= terr_norm
        # cnts['fortify'] /= terr_norm
        # tmp_g.nodes[node]['feat'] = list(cnts.values())

    # for k, v in log.items():
    #     log[k] = (np.array(v) - np.mean(v)) / (np.std(v) + 1e-5)

    # for k, v in log.items():
    #     r = np.max(v) - np.min(v)
    #     if r:
    #         log[k] = (np.array(v) - np.min(v)) / r
    #     else:
    #         log[k] = 0.5 * np.ones(len(v))

    for i, node in enumerate(tmp_g.nodes()):
        tmp_g.nodes[node]['feat'] = [v[i] for v in list(log.values())]

    for edge in tmp_g.edges(data=True):
        edge[2]['feat'] = tmp_g.nodes[edge[0]]['feat'] + tmp_g.nodes[edge[1]]['feat']

    # Prune Leafs
    to_remove = []
    for node in tmp_g.nodes(data='type'):
        if node[1] == 'unit' or node[1] == 'place':  # sum(node[1]) == 0 and node[0] != 'dummy':
            to_remove.append(node[0])
    tmp_g.remove_nodes_from(to_remove)

    # Add Action Indices
    reinforce_index = [n[0] for n in tmp_g.nodes(data=True) if 'player' in n[1] and n[1]['player'] == agent_id]
    ae = board.player_attack_edges(agent_id)
    fe, fw = board.player_fortify_edges(agent_id)
    attack_index = [-1 for _ in ae]
    fortify_index = [-1 for _ in fe]
    move_index = [-1]

    for i, e in enumerate(tmp_g.edges(data=True)):
        if e[2]['type'] == 'attack' and (e[0], e[1]) in ae:
            attack_index[ae.index((e[0], e[1]))] = i
        if e[2]['type'] == 'fortify' and (e[0], e[1]) in fe:
            fortify_index[fe.index((e[0], e[1]))] = i
            if board.last_attack == (e[1], e[0]):
                move_index = [i]
    value_index = [i for i, n in enumerate(tmp_g.nodes(data=True)) if n[1]['type'] == 'player']

    # To Pytorch Geometric
    node_feat = []
    edge_index = []
    edge_feat = []
    node_type = []
    edge_type = []

    for node in tmp_g.nodes(data=True):
        node_feat.append(node[1]['feat'])
        node_type.append(node[1]['type_id'])

    for edge in tmp_g.edges(data=True):
        edge_type.append(edge[2]['type_id'])
        edge_index.append([tmp_g.nodes[edge[0]]['id'], tmp_g.nodes[edge[1]]['id']])
        edge_feat.append(edge[2]['feat'])

    data = Data(x=torch.tensor(node_feat, dtype=torch.float),
                edge_index=torch.tensor(edge_index).t(),
                edge_attr=torch.tensor(edge_feat),
                node_type=torch.tensor(node_type),
                edge_type=torch.tensor(edge_type),
                #####################################
                reinforce_index=torch.tensor(reinforce_index, dtype=torch.long),
                attack_index=torch.tensor(attack_index, dtype=torch.long),
                fortify_index=torch.tensor(fortify_index, dtype=torch.long),
                move_index=torch.tensor(move_index, dtype=torch.long),
                value_index=torch.tensor(value_index, dtype=torch.long),
                ####################################
                task_id=board.state,
                agent_id=agent_id
                )
    # data = T.NormalizeFeatures()(data) if normalize else data
    # data = T.AddSelfLoops()(data) if self_loop else data
    # data = T.ToUndirected()(data) if undirected else data

    return data


def get_hetro_geom_from_board(board, n_agents, normalize=False, self_loop=False, undirected=False):
    tmp_g = nx.MultiDiGraph(board.g)
    to_remove = [edge for edge in tmp_g.edges()]
    tmp_g.remove_edges_from(to_remove)
    for node in tmp_g.nodes(data=True):
        node[1]['type'] = 'country'
        node[1]['id'] = node[0]
        node[1]['label'] = node[0]

    for agent in range(n_agents):
        tmp_g.add_edges_from(board.player_attack_edges(agent), type='attack')
        f_edges, f_weights = board.player_fortify_edges(agent)
        tmp_g.add_edges_from(f_edges, type='fortify')

        tmp_g.add_node('a_{}'.format(agent), type='player', label=agent, id=tmp_g.number_of_nodes())
        for node in tmp_g.nodes(data=True):
            if 'player' in node[1] and node[1]['player'] == agent:
                tmp_g.add_edge('a_{}'.format(agent), node[0], type='own')
    unit_nodes, unit_edges = [], []
    for node in tmp_g.nodes(data=True):
        if node[1]['type'] == 'country':
            unit_nodes += ['u_{}_{}'.format(node[0], u) for u in range(node[1]['units'])]
            unit_edges += [[node[0], 'u_{}_{}'.format(node[0], u)] for u in range(node[1]['units'])]
            unit_edges += [['a_{}'.format(node[1]['player']), 'u_{}_{}'.format(node[0], u)] for u in
                           range(node[1]['units'])]

    tmp_g.add_nodes_from(
        [(un, {'type': 'unit', 'label': i, 'id': tmp_g.number_of_nodes() + i}) for i, un in enumerate(unit_nodes)])
    tmp_g.add_edges_from(unit_edges, type='have')
    draw(tmp_g)

    # Generate Features:
    for node in tmp_g.nodes():
        cnts = {'own': 0, 'have': 0, 'attack': 0, 'fortify': 0}
        for nb in tmp_g.adj[node].items():
            edge_type = nb[1][0]['type']
            cnts[edge_type] += 1.0
        tmp_g.nodes[node]['feat'] = list(cnts.values())
    for edge in tmp_g.edges(data=True):
        edge[2]['feat'] = tmp_g.nodes[edge[0]]['feat'] + tmp_g.nodes[edge[1]]['feat']

    # Prune Leafs
    to_remove = []
    for node in tmp_g.nodes(data='feat'):
        if sum(node[1]) == 0:
            to_remove.append(node[0])
    tmp_g.remove_nodes_from(to_remove)

    # To Pytorch Geometric
    node_feat = {}
    edge_index = {}
    edge_feat = {}
    data = HeteroData()
    for node in tmp_g.nodes(data=True):
        if node[1]['type'] not in node_feat:
            node_feat[node[1]['type']] = []
        node_feat[node[1]['type']].append(node[1]['feat'])
    for edge in tmp_g.edges(data=True):
        t1 = tmp_g.nodes[edge[0]]['type']
        t2 = edge[2]['type']
        t3 = tmp_g.nodes[edge[1]]['type']
        if (t1, t2, t3) not in edge_index:
            edge_index[(t1, t2, t3)] = []
            edge_feat[(t1, t2, t3)] = []
        edge_index[(t1, t2, t3)].append([tmp_g.nodes[edge[0]]['id'], tmp_g.nodes[edge[0]]['id']])
        edge_feat[(t1, t2, t3)].append(edge[2]['feat'])
    for node_type, node_data in node_feat.items():
        data[node_type].x = torch.tensor(node_data)
    for (edge_type, edge_index), edge_feat in zip(edge_index.items(), edge_feat.values()):
        data[edge_type].edge_index = torch.tensor(edge_index).t()
        data[edge_type].edge_attr = torch.tensor(edge_feat)
    data = T.NormalizeFeatures()(data) if normalize else data
    data = T.AddSelfLoops()(data) if self_loop else data
    data = T.ToUndirected()(data) if undirected else data
    return data


def get_feat_adj_type_from_board(board, n_agents, normalized=False):
    feats = []
    types = []
    all_units = sum([a[1] for a in board.g.nodes(data='units')])
    unit_norm = all_units if normalized else 1
    terr_norm = len(board.g.nodes) if normalized else 1
    for node in board.g.nodes(data=True):
        feats.append([len(board.g.adj[node[0]]) / terr_norm, node[1]['units'] / unit_norm])
        types.append(0)

    # feat_player = []
    for p in range(n_agents):
        feats.append([len(board.player_nodes(p)) / terr_norm, board.player_units(p) / unit_norm])
        types.append(1)

    temp_g = nx.Graph(deepcopy(board.g))
    temp_g.add_nodes_from(['p' + str(i) for i in range(n_agents)])
    edges = [[node[0], 'p' + str(node[1]['player'])] for node in temp_g.nodes(data=True) if 'player' in node[1]]
    temp_g.add_edges_from(edges)

    adj = nx.adjacency_matrix(temp_g).todense() + np.eye(len(temp_g.nodes()))
    return feats, adj, types


def get_feat_adj_from_board(board, player, n_agents, n_grps):
    feats = []
    all_units = sum([a[1] for a in board.g.nodes(data='units')])
    for node in board.g.nodes(data=True):
        feats.append([
            0,
            (1 if player == node[1]['player'] else -1) * node[1]['units'] / all_units,
            -1,  # node[1]['gid'],
            0,  # board.info['group_reward'][str(node[1]['gid'])],
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

    temp_g.add_edges_from(edges)

    feats = [list(flatten([to_one_hot(feat[0], 2), feat[1], to_one_hot(feat[2], n_grps), feat[3],
                           to_one_hot(feat[4], 2), to_one_hot(feat[5], 2)])) for feat in feats]
    adj = nx.adjacency_matrix(temp_g).todense() + np.eye(len(temp_g.nodes()))
    return feats, adj
