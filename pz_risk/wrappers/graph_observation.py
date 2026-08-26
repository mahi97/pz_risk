from gymnasium.spaces import Box, Discrete, Dict, MultiBinary
from pettingzoo.utils.wrappers import BaseWrapper

from pz_risk.core.gamestate import GameState
from pz_risk.utils import get_feat_adj_from_board


class GraphObservationWrapper(BaseWrapper):
    """
    this wrapper crashes for invalid Risk! actions
    # Should be used for Discrete spaces
    """

    def __init__(self, env):
        super().__init__(env)
        self.n_nodes = env.unwrapped.n_nodes
        self.n_grps = env.unwrapped.n_grps
        self.n_agents = env.unwrapped.n_agents
        self.observation_spaces = Dict({
            'adj': MultiBinary((self.n_agents + self.n_nodes) ** 2),
            # Node{card?, #unit, grp_id, grp_p} + Player{cards?, me?}
            'feat': Box(0, 1000, shape=[2 + 1 + self.n_grps + 1 + 2 + 2]),
            'task_id': Discrete(len(GameState))
        })
        self.action_spaces = self.env.action_spaces

    def observe(self, agent):
        board = self.env.observe(agent)
        feats, adj = get_feat_adj_from_board(board, agent, self.n_agents, self.n_grps)
        return {
            'adj': adj,
            'feat': feats,
            'task_id': board.state
        }

    def __str__(self):
        return str(self.env)
