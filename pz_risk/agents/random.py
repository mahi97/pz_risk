from agents.base import BaseAgent
from agents.sampling import SAMPLING


class RandomAgent(BaseAgent):
    def __init__(self, player_id):
        super(RandomAgent, self).__init__()
        self.player_id = player_id

    def reset(self):
        pass

    def act(self, state):
        return SAMPLING[state['game_state']](state['board'], self.player_id)
