import numpy as np

from agents.base import BaseAgent
from core.gamestate import GameState
from agents.sampling import SAMPLING
from agents.value import manual_advantage, manual_q

from loguru import logger

class GreedyAgent(BaseAgent):
    def __init__(self, player_id):
        super(GreedyAgent, self).__init__()
        self.player_id = player_id

    def reset(self):
        pass

    def act(self, state):
        if state['game_state'] == GameState.Attack:
            attack_edge = state['board'].player_attack_edges(self.player_id)
            base = manual_advantage(state, self.player_id, (1, None))  # Attack Finished
            v = [manual_advantage(state, self.player_id, (False, ae)) for ae in attack_edge]
            edge = attack_edge[np.argmax(v)]
            logger.info('Attack values:{}, base: {}'.format(v, base))
            return (1, (None, None)) if base > max(v) else (0, edge)
        else:
            return SAMPLING[state['game_state']](state['board'], self.player_id)
        # Find the action with highest advantage
        # Execute the action
