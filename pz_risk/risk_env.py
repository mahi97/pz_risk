from gym.spaces import Discrete, MultiDiscrete, Tuple, Dict
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers

import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from board import Board, BOARDS
NUM_ITERS = 100
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

        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        self.action_spaces = {agent: {'reinforce': MultiDiscrete([n_nodes, 100]),
                                      'attack': Discrete(n_edges),
                                      'fortify': MultiDiscrete([n_nodes, n_nodes, 100])
                                      } for agent in self.possible_agents}
        self.observation_spaces = {agent: Tuple(tuple([MultiDiscrete([n_agent, 100]) for _ in range(n_nodes)]))
                                   for agent in self.possible_agents}
        self.board.reset(n_agent, 20, 7)

    def render(self, mode="human"):
        '''
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        '''
        G = self.board.g
        # pos = nx.spring_layout(G, seed=3113794652)  # positions for all nodes
        # pos = nx.kamada_kawai_layout(G)  # positions for all nodes
        pos = {i+1: p for i, p in enumerate(self.board.pos)} if self.board.pos is not None else nx.kamada_kawai_layout(G)
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
        return np.array(self.observations[agent])

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
        # self.state = {agent: NONE for agent in self.agents}
        # self.observations = {agent: NONE for agent in self.agents}
        self.num_moves = 0
        '''
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        '''
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

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

        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        self._cumulative_rewards[agent] = 0

        # stores action of current agent
        self.state[self.agent_selection] = action

        # collect reward if it is the last agent to act
        if self._agent_selector.is_last():
            # rewards for all agents are placed in the .rewards dictionary
            self.rewards[self.agents[0]], self.rewards[self.agents[1]] = REWARD_MAP[(self.state[self.agents[0]], self.state[self.agents[1]])]

            self.num_moves += 1
            # The dones dictionary must be updated for all players.
            self.dones = {agent: self.num_moves >= NUM_ITERS for agent in self.agents}

            # observe the current state
            for i in self.agents:
                self.observations[i] = self.state[self.agents[1 - self.agent_name_mapping[i]]]
        else:
            # necessary so that observe() returns a reasonable observation at all times.
            self.state[self.agents[1 - self.agent_name_mapping[agent]]] = NONE
            # no rewards are allocated until both players give an action
            self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

if __name__ == '__main__':
    e = env()
    e.reset()
    e.render()