import numpy as np
from gymnasium.spaces import Box
from pettingzoo.utils.wrappers import BaseWrapper


class VectorObservationWrapper(BaseWrapper):
    """
    this wrapper crashes for invalid Risk! actions
    # Should be used for Discrete spaces
    """

    def __init__(self, env):
        super().__init__(env)
        self.n_nodes = env.unwrapped.n_nodes
        self.n_grps = env.unwrapped.n_grps
        self.n_agents = env.unwrapped.n_agents
        self.obs_shape = [self.n_nodes, self.n_agents + self.n_grps + 1 + 1 + 1]
        self.observation_spaces = Box(0, 1000, self.obs_shape)
        self.action_spaces = self.env.action_spaces

    def observe(self, agent):
        board = self.env.observe(agent)
        obs = np.zeros(self.obs_shape)
        for node in board.g.nodes():
            pass
        return obs

    def __str__(self):
        return str(self.env)
