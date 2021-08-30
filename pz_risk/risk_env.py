import math
import random

from gym.spaces import Discrete, MultiDiscrete, Tuple, Dict, Box
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers

import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from typing import Tuple

from board import Board, BOARDS, GameState

NUM_ITERS = 100
MAX_CARD = 10
MAX_UNIT = 100
COLORS = [
    'tab:red',
    'tab:blue',
    'tab:green',
    'tab:purple',
    'tab:pink',
    'tab:cyan',
]


def env():
    '''
    The env function wraps the environment in 3 wrappers by default. These
    wrappers contain logic that is common to many pettingzoo environments.
    We recommend you use at least the OrderEnforcingWrapper on your own environment
    to provide sane error messages. You can find full documentation for these methods
    elsewhere in the developer documentation.
    '''
    env = raw_env()
    env = wrappers.CaptureStdoutWrapper(env)
    # env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv):
    '''
    The metadata holds environment constants. From gym, we inherit the "render.modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    '''
    metadata = {'render.modes': ['human'], "name": "rps_v2"}

    def __init__(self, n_agent=6, map='world'):
        '''
        - n_agent: Number of Agent
        - board: ['test', 'world', 'world2']
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces

        These attributes should not be changed after initialization.
        '''
        super().__init__()
        self.board = BOARDS[map]
        n_nodes = self.board.g.number_of_nodes()
        n_edges = self.board.g.number_of_edges()
        self.possible_agents = [r for r in range(n_agent)]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.reinforce_cst = []
        self.fortify_cst = []

        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        self.action_spaces = {agent: {GameState.Reinforce: MultiDiscrete([n_nodes, 100]),
                                      GameState.Attack: Discrete(n_edges),
                                      GameState.Fortify: MultiDiscrete([n_nodes, n_nodes, 100]),
                                      GameState.Start: Discrete(1),
                                      GameState.End: Discrete(1),
                                      GameState.Card: Discrete(2),
                                      GameState.Move: Discrete(100)
                                      } for agent in self.possible_agents}
        self.observation_spaces = {agent: Discrete(MAX_UNIT) for agent in self.possible_agents}  # placement
        self.observation_spaces['board'] = Dict({})
        self.observation_spaces['cards'] = MultiDiscrete([MAX_CARD for _ in range(n_agent)])
        self.observation_spaces['my_cards'] = Discrete(2)

    def sample(self):
        p = self.agent_selection
        u = self.placement[p]
        if self.state_selection == GameState.Reinforce:
            nodes = self.board.player_nodes(p)
            branches = math.comb(u + len(nodes)- 1, u)
            index = np.random.randint(branches)
            r = sorted([(index // u**i) % u  for i in range(len(nodes))])
            r.append(u)
            r2 = {n:r[i+1] - r[i] for i, n in zip(range(len(nodes)), random.sample(nodes, len(nodes)))}
            return [(0 if n not in nodes else r2[n]) for n in self.board.g.nodes()]
        elif self.state_selection == GameState.Card:
            return np.random.randint(2)
        elif self.state_selection == GameState.Attack:
            edges = self.board.player_attack_edges(p)
            index = np.random.randint(len(edges))
            return edges[index]
        elif self.state_selection == GameState.Move:
            return np.random.randint(u)
        elif self.state_selection == GameState.Fortify:
            cc = self.board.player_connected_components(p)
            branches = []
            potential_sources = []
            for c in cc:
                potential_source = [node for node in c if self.board.g.nodes[node]['units'] >= 2]
                branches.append(len(potential_source) * (len(c) - 1))
                potential_sources.append(potential_source)
            index = np.random.randint(branches)
            total_branch = 0
            src = -1
            trg = -1
            for b, c, ps in zip(branches, cc, potential_sources):
                total_branch += b
                if index < total_branch:
                    src = ps[index // len(c) - 1]
                    trg = c[index % len(c)] if index % len(c) < index // len(c) else c[index % len(c) + 1]
            num_unit = np.random.randint(self.board.g.nodes[src]['units'])
            return src, trg, num_unit


    def render(self, mode="human"):
        '''
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        '''
        G = self.board.g
        # pos = nx.spring_layout(G, seed=3113794652)  # positions for all nodes
        # pos = nx.kamada_kawai_layout(G)  # positions for all nodes
        pos = {i + 1: p for i, p in
               enumerate(self.board.pos)} if self.board.pos is not None else nx.kamada_kawai_layout(G)
        if mode == 'human':
            options = {"edgecolors": "tab:gray", "node_size": 800, "alpha": 0.9}
            for agent in self.agents:
                nx.draw_networkx_nodes(G, pos,
                                       nodelist=[c[0] for c in G.nodes(data=True) if c[1]['player'] == agent],
                                       node_color=COLORS[agent], **options)
            # nx.draw_networkx_nodes(G, pos, nodelist=[4, 5, 6, 7], node_color="tab:blue", **options)

            # edges
            nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

            # some math labels
            labels = {c[0]: c[1]['units'] for c in G.nodes(data=True)}
            nx.draw_networkx_labels(G, pos, labels, font_size=22, font_color="black")

            plt.tight_layout()
            plt.axis("off")
            plt.show()
        else:
            print('Wait for it')

    def observe(self, agent):
        '''
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        '''
        # observation of one agent is the previous state of the other
        return {'board': self.board.g,
                'my_card': self.board.player[agent].cards,
                'placement': self.placement[agent],
                'game_state': self.state_selection,
                'cards': [len(p.cards) for p in self.board.player]}

    def close(self):
        '''
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        '''
        pass

    def reset(self):
        '''
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - dones
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.

        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        '''
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.num_moves = 0
        self.placement = {agent: 0 for agent in self.agents}
        self.board.reset(len(self.agents), 20, 7)
        '''
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        '''
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.state_selection = GameState.Start
        print(self.agent_selection, self.state_selection)

    def reward(self, agent):
        return 0.0

    def done(self, agent):
        return False

    def step(self, action):
        '''
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - dones
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        '''
        if self.dones[self.agent_selection]:
            # handles stepping an agent which is already done
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next done agent,  or if there are no more done agents, to the next live agent
            return self._was_done_step(action)

        agent = self.agent_selection
        state = self.state_selection

        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        self._cumulative_rewards[agent] = 0

        # TODO: Check action before sending to board

        # stores action of current agent
        self.board.step(agent, state, action)

        # collect reward if it is the last agent to act
        if self._agent_selector.is_last() and self._state_selector.is_last():
            # rewards for all agents are placed in the .rewards dictionary

            self.rewards = {agent: self.reward(agent) for agent in self.agents}

            self.num_moves += 1
            # The dones dictionary must be updated for all players.
            self.dones = {agent: self.done(agent) for agent in self.agents}

        else:
            # no rewards are allocated until both players give an action
            self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        self.state_selection = GameState.Start # TODO: ...
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()


if __name__ == '__main__':
    e = env()
    e.reset()
    e.render()
    for agent in e.agent_iter():
        obs, rew, done, info = e.last()
        action_space = e.action_spaces[agent][obs['game_state']]
        e.step(e.unwrapped.sample())
