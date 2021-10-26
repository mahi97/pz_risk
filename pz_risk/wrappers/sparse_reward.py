from pettingzoo.utils.wrappers import BaseWrapper
from core.gamestate import GameState


class SparseRewardWrapper(BaseWrapper):
    """
    this wrapper crashes for invalid Risk! actions
    # Should be used for Discrete spaces
    """

    def __init__(self, env):
        super().__init__(env)
        self.board = env.unwrapped.board
        self.n_nodes = env.unwrapped.n_nodes
        self.n_agents = env.unwrapped.n_agents
        self.time = 0

    def reset(self):
        super(SparseRewardWrapper, self).reset()
        self.time = 0

    def reward(self, agent):
        rew = 2*len(self.board.player_nodes(agent)) - self.n_nodes
        # rew -= self.time / 1000
        if len(self.board.player_nodes(agent)) == self.n_nodes:
            rew = 1000 - self.time / 100
        elif len(self.board.player_nodes(agent)) == 0:
            rew = -1000 + self.time / 100
        return rew

    def done(self, agent):
        return len(self.board.player_nodes(agent)) == 0 or len(self.board.player_nodes(agent)) == self.n_nodes

    def observe(self, agent):
        obs = super().observe(agent)
        obs['rewards'] = [self.reward(i) for i in range(self.n_agents)]
        obs['dones'] = [self.done(i) for i in range(self.n_agents)]
        self.time += 1
        return obs

    def __str__(self):
        return str(self.env)
