from pettingzoo.utils.wrappers import BaseWrapper
from core.gamestate import GameState
from multiprocessing import Pool
import utils


class VECIterable:
    def __init__(self, envs, max_iter):
        self.envs = envs
        self.max_iter = max_iter

    def __iter__(self):
        return VECIterator(self.envs, self.max_iter)


class VECIterator:
    def __init__(self, v_env, max_iter):
        self.v_env = v_env
        self.iters_til_term = [max_iter for e in self.v_env.envs]

    def __next__(self):
        agent_selections = []
        if max(self.iters_til_term) <= 0:
            raise StopIteration
        for i, env in enumerate(self.v_env.envs):
            if self.iters_til_term[i] <= 0:
                agent_selections.append(-1)
                continue
            self.iters_til_term[i] -= 1
            agent_selections.append(self.v_env.agent_selections[i])
        return agent_selections


class VectorizeWrapper:
    """
    this wrapper crashes for invalid Risk! actions
    # Should be used for Discrete spaces
    """

    def __init__(self, envs):
        self.envs = envs
        self.skip = []
        self.exec_pool = Pool(len(self.envs))

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['exec_pool']
        return self_dict

    @property
    def unwrapped(self):
        return [env.unwrapped for env in self.envs]

    def seed(self, seed=None):
        [env.seed(seed) for env in self.envs]

    def close(self):
        [env.close() for env in self.envs]

    def render(self, mode="human"):
        return self.envs[0].render(mode)

    def _reset(self, e):
        e.reset()
        return e

    def reset(self):
        data = self.exec_pool.map(self._reset, self.envs)
        self.envs = data
        print('order: ', [e.unwrapped.id for e in self.envs])
        self.agent_selections = [e.agent_selection for e in self.envs]
        self.rewards = [e.rewards for e in self.envs]
        self.dones = [e.dones for e in self.envs]
        self.infos = [e.infos for e in self.envs]
        self.agents = [e.agents for e in self.envs]
        self._cumulative_rewards = [e._cumulative_rewards for e in self.envs]

    def reward(self, agent, last=False):
        return [e.reward(agent, last) for e in self.envs]

    def observe(self, agents):
        return [env.observe(agent) for env, agent in zip(self.envs, agents)]

    def state(self):
        return [env.state() for env in self.envs]

    def _step(self, e_a):
        env, action = e_a
        env.step(action)
        return env

    def step(self, actions):
        envs = [self.envs]
        envs = self.exec_pool.map(self._step, zip(self.envs, actions))

        self.agent_selections = [e.agent_selection for e in envs]
        self.rewards = [e.rewards for e in envs]
        self.dones = [e.dones for e in envs]
        self.infos = [e.infos for e in envs]
        self.agents = [e.agents for e in envs]
        self._cumulative_rewards = [e._cumulative_rewards for e in envs]

    def __str__(self):
        return str(self.envs[0])

    def agent_iter(self, max_iter=2 ** 63):
        """
        yields the current agent (self.agent_selection) when used in a loop where you step() each iteration.
        """
        return VECIterable(self, max_iter)

    def _last(self, e_a):
        e, a = e_a
        obs = e.observe(a)
        return e, a, obs

    def last(self, observe=True):
        """
        returns observation, cumulative reward, done, info   for the current agent (specified by self.agent_selection)
        """
        obs, rew, done, info = [], [], [], []
        data = self.exec_pool.map(self._last, zip(self.envs, self.agent_selections))
        envs = []
        for e, a, observation in data:
            obs.append(observation)
            rew.append(e._cumulative_rewards[a])
            done.append(e.dones[a])
            info.append(e.infos[a])
            envs.append(e)
        self.envs = envs
        return obs, rew, done, info

    def _valid_action(self, b_p):
        board, player = b_p
        return board.valid_actions(player)[1]

    def valid_actions(self, agent_ids):
        valid_acts = self.exec_pool.map(self._valid_action, zip([e.board for e in self.envs], agent_ids))
        return valid_acts

    def get_data(self, agent_ids, n_agents):
        single_arg = []
        for b, i in zip([e.board for e in self.envs], agent_ids):
            single_arg.append([b, n_agents, i])
        data = self.exec_pool.map(utils.get_geom_from_board_single_arg, single_arg)
        return data
