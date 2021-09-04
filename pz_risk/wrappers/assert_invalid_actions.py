from pettingzoo.utils.wrappers import BaseWrapper
from core.gamestate import GameState


class AssertInvalidActionsWrapper(BaseWrapper):
    """
    this wrapper crashes for invalid Risk! actions
    # Should be used for Discrete spaces
    """

    def __init__(self, env):
        super().__init__(env)
        self.board = env.unwrapped.board

    def step(self, action):
        player = self.agent_selection
        state = self.board.state
        u = self.board.players[player].placement
        gn = lambda x: self.board.g[x]['name']
        if state == GameState.Reinforce:
            assert sum(action) == u, 'sum(action) != player placement: {} != {}'.format(sum(action), u)
            assert min(action) >= 0, 'min(action) is less than zero! {}'.format(min(action))
            for node, units in enumerate(action):
                assert units <= 0 or node + 1 in self.board.player_nodes(player), \
                    'selected node is not owned by player: node: {}, player: {}'.format(self.board.g.nodes[node + 1]['name'], player)
        elif state == GameState.Card:
            assert 0 <= action <= 1, 'Card Action should be 0 or 1: {}'.format(action)
        elif state == GameState.Attack:
            edges = self.board.player_attack_edges(player)
            assert action[0] <= 1, 'Attack Finished should be 0 or 1: {}'.format(action[0])
            assert action[1] in edges, 'Attack Can not be performed from {} to {}'.format(gn(action[1][0]), gn(action[1][1]))
        elif state == GameState.Move:
            u = max(0, self.board.g.nodes[self.board.last_attack[1]]['units'] - 3)
            assert 0 <= action <= u, 'Move out of bound: {} ~ {}'.format(action, u)
        elif state == GameState.Fortify:
            cc = self.board.player_connected_components(player)
            c = [c for c in cc if action[1] in c][0]
            assert 0 <= action[0] <= 1, 'Skip should be 0 or 1: {}'.format(action[0])
            assert action[2] in c, 'Fortify Can not be performed from {} to {}'.format(gn(action[1]), gn(action[2]))
            assert action[3] <= self.board.g.nodes[action[1]]['units'] ,'Fortify Can not be more than source units!'

        super().step(action)

    def __str__(self):
        return str(self.env)
