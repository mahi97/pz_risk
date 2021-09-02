import math
import random

from gym.spaces import Discrete, MultiDiscrete, Tuple, Dict, Box
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers

import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

from core.board import Board, BOARDS
from core.gamestate import GameState

from loguru import logger

from utils import *
from agents.sampling import SAMPLING

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
    """
    The env function wraps the environment in 3 wrappers by default. These
    wrappers contain logic that is common to many pettingzoo environments.
    We recommend you use at least the OrderEnforcingWrapper on your own environment
    to provide sane error messages. You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    env = raw_env()
    env = wrappers.CaptureStdoutWrapper(env)
    # env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv):
    """
    The metadata holds environment constants. From gym, we inherit the "render.modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    """
    metadata = {'render.modes': ['human'], "name": "rps_v2"}

    def __init__(self, n_agent=6, board_name='world'):
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
        self.board = BOARDS[board_name]
        n_nodes = self.board.g.number_of_nodes()
        n_edges = self.board.g.number_of_edges()
        self.possible_agents = [r for r in range(n_agent)]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))

        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        self.action_spaces = {agent: {GameState.Reinforce: MultiDiscrete([n_nodes, 100]),
                                      GameState.Attack: MultiDiscrete([2, n_edges]),  # +1 for Skip
                                      GameState.Fortify: MultiDiscrete([2, n_nodes, n_nodes, 100]),  # Last dim for Skip
                                      GameState.StartTurn: Discrete(1),
                                      GameState.EndTurn: Discrete(1),
                                      GameState.Card: Discrete(2),
                                      GameState.Move: Discrete(100)
                                      } for agent in self.possible_agents}
        self.observation_spaces = {agent: Discrete(MAX_UNIT) for agent in self.possible_agents}  # placement
        self.observation_spaces['board'] = Dict({})
        self.observation_spaces['cards'] = MultiDiscrete([MAX_CARD for _ in range(n_agent)])
        self.observation_spaces['my_cards'] = Discrete(2)

        self.agents = []
        self.rewards = {}
        self._cumulative_rewards = {}
        self.dones = {}
        self.infos = {}
        self.num_turns = 0
        self.placement = {}
        self.num_moves = 1

        self.land_hist = {a: [] for a in self.possible_agents}
        self.unit_hist = {a: [] for a in self.possible_agents}
        self.place_hist = {a: [] for a in self.possible_agents}

    def sample(self):
        return SAMPLING[self.board.state](self.board, self.agent_selection)

    def render_info(self, mode="human"):
        fig = plt.figure(1, figsize=(7, 7))
        plt.clf()

        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        for a in self.possible_agents:
            self.land_hist[a].append(len(self.board.player_nodes(a)))
            self.unit_hist[a].append(self.board.player_units(a))
            self.place_hist[a].append(self.board.players[a].placement)

        for a in self.possible_agents:
            ax1.plot(self.land_hist[a], COLORS[a])
            ax1.set_title('# of Lands')
            ax2.plot(self.unit_hist[a], COLORS[a])
            ax2.set_title('# of Units')
            ax3.plot(self.place_hist[a], COLORS[a])
            ax3.set_title('# of Reinforce')
            player_circle = [plt.Circle((0.12 + i*0.15, 0.8), radius=0.12 if a == self.agent_selection else 0.1, color=COLORS[a]) for i, a in enumerate(self.agents)]
            [ax4.add_patch(patch) for patch in player_circle]
            game_rect = [(plt.Rectangle((-0.15 + i * 0.15, 0.3), width=0.1, height=0.3, color='green'), a) for i, a in enumerate(GameState)]
            [ax4.add_patch(patch) for patch, state in game_rect if state.value <= self.board.state.value]
            ax4.text(0, 0.1, 'Game State: {}'.format(self.board.state.name), fontsize=15)
            # ax4.plot(x, -y, 'tab:red')
            # ax4.set_title('Axis [1, 1]')

        # for ax in axs.flat:
        #     ax.set(xlabel='x-label', ylabel='y-label')
        #
        # # Hide x labels and tick labels for top plots and y ticks for right plots.
        # for ax in axs.flat:
        #     ax.label_outer()
        # plt.text(-1, 1,
        #          r'Current Player: {}: {}'.format(self.agent_selection, COLORS[self.agent_selection][4:].title()),
        #          fontsize=10)
        # plt.text(-1, 1, r'Current Player: {}: {}'.format(self.agent_selection, COLORS[self.agent_selection][4:].title()),
        #          fontsize=10)
        plt.tight_layout()
        plt.axis("off")
        plt.pause(0.001)

    def render(self, mode="human"):
        '''
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        '''
        plt.figure(0)
        plt.clf()

        G = self.board.g
        # pos = nx.spring_layout(G, seed=3113794652)  # positions for all nodes
        # pos = nx.kamada_kawai_layout(G)  # positions for all nodes
        pos = {i: n['pos'] for i, n in self.board.g.nodes(data=True)}
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
            plt.pause(0.001)
            self.render_info()
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
                'my_card': self.board.players[agent].cards,
                'placement': self.board.players[agent].placement,
                'game_state': self.board.state,
                'cards': [len(p.cards) for p in self.board.players]}

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        plt.close()

    def reset(self):
        """
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
        """
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.board.reset(len(self.agents), 20, 7)
        self.num_turns = 0
        self.num_moves = 1
        '''
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        '''
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def reward(self, agent):
        return 0.0

    def done(self, agent):
        return False

    def validate_action(self, player, state, action):
        u = self.board.players[player].placement
        gn = lambda x: self.board.g[x]['name']
        if state == GameState.Reinforce:
            if sum(action) != u:
                logger.error('sum(action) != player placement: {} != {}'.format(sum(action), u))
                return False
            if min(action) < 0:
                logger.error('min(action) is less than zero! {}'.format(min(action)))
                return False
            for node, units in enumerate(action):
                if units > 0 and node + 1 not in self.board.player_nodes(agent):
                    logger.error('selected node is not owned by player: node: {}, player: {}'.format(
                        self.board.g.nodes[node + 1]['name'], player))
                    return False
        elif self.board.state == GameState.Card:
            return 0 <= action <= 1
        elif self.board.state == GameState.Attack:
            edges = self.board.player_attack_edges(player)
            if action[0] > 1:
                logger.error('Attack Finished should be 0 or 1: {}'.format(action[0]))
                return False
            if action[1] not in edges:
                logger.error('Attack Can not be performed from {} to {}'.format(gn(action[1][0]), gn(action[1][1])))
                return False
        elif self.board.state == GameState.Move:
            u = max(0, self.board.g.nodes[self.board.last_attack[1]]['units'] - 3)
            if action < 0 or action > u:
                logger.error('Move out of bound: {} ~ {}'.format(action, u))
                return False
        elif self.board.state == GameState.Fortify:
            cc = self.board.player_connected_components(player)
            c = [c for c in cc if action[1] in c][0]
            if action[0] > 1:
                logger.error('Skip should be 0 or 1: {}'.format(action[0]))
                return False
            if action[2] not in c:
                logger.error('Fortify Can not be performed from {} to {}'.format(gn(action[1]), gn(action[2])))
                return False
            if action[3] > self.board.g.nodes[action[1]]['units']:
                logger.error('Fortify Can not be more than source units!')
                return False
        return True

    def step(self, action):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - dones
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        if self.dones[self.agent_selection]:
            # handles stepping an agent which is already done
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next done agent,  or if there are no more done agents, to the next live agent
            return self._was_done_step(action)

        agent = self.agent_selection
        state = self.board.state
        logger.info('Player: {}, State: {}, Actions: {}'.format(agent, state, action))

        self._cumulative_rewards[agent] = 0

        valid = self.validate_action(agent, state, action)
        if not valid:
            logger.error('Action is not valid! {}'.format(action))
            exit(1)

        self.board.step(agent, action)

        # collect reward if it is the last agent to act
        if self._agent_selector.is_last() and self.board.state == GameState.StartTurn:
            # rewards for all agents are placed in the .rewards dictionary

            self.rewards = {agent: self.reward(agent) for agent in self.agents}

            self.num_turns += 1
            # The dones dictionary must be updated for all players.
            self.dones = {agent: self.done(agent) for agent in self.agents}

        else:
            # no rewards are allocated until both players give an action
            self._clear_rewards()

        # selects the next agent.
        if self.board.state == GameState.StartTurn:
            self.num_moves += 1
            self.agent_selection = self._agent_selector.next()
            while len(self.board.player_nodes(self.agent_selection)) == 0:
                self.dones[self.agent_selection] = True
                self.agent_selection = self._agent_selector.next()
            self.board.step(self.agent_selection, None)
        if self.board.state == GameState.EndTurn:
            self.dones = {agent: True for agent in self.agents}
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()


if __name__ == '__main__':
    e = env()
    e.reset()
    # e.render()
    winner = -1
    for agent in e.agent_iter():
        obs, rew, done, info = e.last()
        if done:
            continue
        e.step(e.unwrapped.sample())
        if all(e.dones.values()):
            winner = agent
            break
        e.render()
    # e.render()
    plt.show()
    logger.info('Done in {} Turns and {} Moves. Winner is Player {}'
                .format(e.unwrapped.num_turns, e.unwrapped.num_moves, winner))
