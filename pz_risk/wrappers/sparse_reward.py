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
        self.id = env.unwrapped.id
        self.time = 0

    def reset(self):
        super(SparseRewardWrapper, self).reset()
        self.agents = self.env.agents
        self.time = 0

    def reward(self, agent, last=False):
        rew = 0.0
        # rew -= 0.01
        # rew -= self.time / 2000
        # rew = -0.75 if rew < -0.75 else rew
        if last:
            rew = (2*len(self.board.player_nodes(agent)) - self.n_nodes) / self.n_nodes
            return rew  # - 0.2
        if len(self.board.player_nodes(agent)) == self.n_nodes:
            rew = 1.0
        elif len(self.board.player_nodes(agent)) == 0:
            rew = -1.0
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
