import numpy as np
from copy import deepcopy
import torch
import math
from core.gamestate import GameState

EPS = 1e-8


# from multiprocessing import Pool as CPool
#
# e = CPool()
class MCTS:
    def __init__(self, n_agent, args):
        self.cpuct = 1
        # self.board = deepcopy(board)
        # self.critic = critic
        self.args = args
        self.n_agent = n_agent
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        # self.Vs = {}  # stores game.getValidMoves for board s
        # self.exec_pool = CPool()

    def reset(self):
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        # self.Vs = {}  # stores game.getValidMoves for board s

    def getActionProb(self, board, player, critic, step=0, pure=False, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        b = deepcopy(board)
        s = board.to_string(player)

        if pure:
            # global e
            # e.map(self.pure_search, [(b, player)] * 10)
            for i in range(self.args.mcts):
                self.pure_search(b, player, step)
        else:
            for i in range(self.args.mcts):
                self.search(b, player, critic, step)

        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(len(board.valid_actions(player)[1]))]

        if temp == 0 or sum(counts) == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def pure_search(self, board, player, step):
        if board.state == GameState.StartTurn:
            board.step(player, None)
            player = (player + 1) % self.n_agent
            return self.pure_search(board, player, step + 1)

        s = board.to_string(player)

        if s not in self.Es:
            self.Es[s] = board.game_ended(player, self.args.num_steps)
        if self.Es[s] is not None or step > self.args.num_steps:
            # terminal node
            return self.Es[s]

        size = len(board.valid_actions(player)[1])
        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = [1 / size for _ in range(size)], 0.0
            self.Ns[s] = 0
            return v

        # valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(len(board.valid_actions(player)[1])):
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
            else:
                u = self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

            if u > cur_best:
                cur_best = u
                best_act = a

        a = best_act
        board.step(player, board.valid_actions(player)[1][a])
        # player = (player + 1) % 2  # len(2)

        next_s = board  # self.game.getCanonicalForm(next_s, next_player)

        v = self.pure_search(next_s, player, step+1)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return v

    def search(self, board, player, critic, step):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        if board.state == GameState.StartTurn:
            board.step(player, None)
            player = (player + 1) % 2
            return self.pure_search(board, player, step+1)

        s = board.to_string(player)

        if s not in self.Es:
            self.Es[s] = board.game_ended(player, self.args.num_steps)
        if self.Es[s] is not None or step > self.args.num_steps:
            # terminal node
            return self.Es[s]

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = critic.predict(board, player)
            self.Ns[s] = 0
            return v

        # valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(len(board.valid_actions(player)[1])):
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
            else:
                u = self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

            if u > cur_best:
                cur_best = u
                best_act = a

        a = best_act
        board.step(player, board.valid_actions(player)[1][a])
        player = (player + 1) % 2  # len(2)

        next_s = board  # self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s, player, critic, step+1)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return v
