from pettingzoo.utils.wrappers import BaseWrapper
from core.gamestate import GameState
from agents.value import manual_value


class DenseRewardWrapper(BaseWrapper):
    """
    this wrapper crashes for invalid Risk! actions
    # Should be used for Discrete spaces
    """

    def __init__(self, env):
        super().__init__(env)
        self.board = env.unwrapped.board
        self.n_agents = env.unwrapped.n_agents
        # self.cum_rew = 0

    def reset(self):
        super(DenseRewardWrapper, self).reset()
        # self.cum_rew = 0

    def reward(self, agent):
        return manual_value(self.board, agent)
        # return self.cum_rew

    def done(self, agent):
        return len(self.board.player_nodes(agent)) == 0

    def observe(self, agent):
        obs = super().observe(agent)
        obs['rewards'] = [self.reward(i) for i in range(self.n_agents)]
        obs['dones'] = [self.done(i) for i in range(self.n_agents)]
        return obs

    def __str__(self):
        return str(self.env)
