import numpy as np

from core.board import Board
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

    def act(self, state: Board):
        # if state.state == GameState.Attack:
        #     attack_edge = state.player_attack_edges(self.player_id)
        #     base = manual_advantage(state, self.player_id, (1, None))  # Attack Finished
        #     v = [manual_advantage(state, self.player_id, (False, ae)) for ae in attack_edge]
        #     edge = attack_edge[np.argmax(v)]
        #     # logger.info('Attack values:{}, base: {}'.format(v, base))
        #     return (1, (None, None)) if base > max(v) else (0, edge)
        # else:
            # Use Model to Gather Future State per Valid Actions
        action_scores = []
        deterministic, valid_actions = state.valid_actions(self.player_id)
        for valid_action in valid_actions:
            action_scores.append(manual_advantage(state, self.player_id, valid_action))
        action = valid_actions[np.argmax(action_scores)]
        return action
        # Find the action with highest advantage
        # Execute the action
