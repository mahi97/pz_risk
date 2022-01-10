from pettingzoo.utils.wrappers import BaseWrapper
from core.gamestate import GameState
import networkx as nx
from copy import deepcopy
from gym.spaces import Discrete, Dict, MultiBinary, Box, MultiDiscrete
from utils import *

from torch_geometric.data import Data


class GeometricObservationWrapper(BaseWrapper):
    """
    this wrapper crashes for invalid Risk! actions
    # Should be used for Discrete spaces
    """

    def __init__(self, env):
        super().__init__(env)
        self.n_nodes = env.unwrapped.n_nodes
        # self.n_grps = env.unwrapped.n_grps
        self.n_agents = env.unwrapped.n_agents
        self.observation_spaces = Dict({
            'adj': MultiBinary((self.n_agents + self.n_nodes) ** 2),
            'feat': Box(0, 1000, shape=[2]),  # Territory, Units
            'type': MultiDiscrete([2 for _ in range(self.n_agents + self.n_nodes)]),
            'task_id': Discrete(len(GameState))
        })
        self.action_spaces = self.env.action_spaces

    def observe(self, agent):
        board = self.env.observe(agent)
        data = get_geom_from_board(board, self.n_agents, agent, True)
        return {
            'data': data,
            'task_id': board.state
        }

    def reset(self):
        super(GeometricObservationWrapper, self).reset()
        self.agents = self.env.agents


    def __str__(self):
        return str(self.env)
