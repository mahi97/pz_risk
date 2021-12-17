from agents.base import BaseAgent
from agents.sampling import SAMPLING


class MCTSAgent(BaseAgent):
    def __init__(self, player_id):
        super(MCTSAgent, self).__init__()
        self.player_id = player_id

    def reset(self):
        pass

    def act(self, state):
        return SAMPLING[state.state](state, self.player_id)
