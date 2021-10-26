# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail

import numpy as np

from loguru import logger

import torch as t
import torch.nn as nn
import torch.functional as F

from training.distributions import Bernoulli, Categorical, DiagGaussian
from training.utils import *


class NNBase(nn.Module):
    def __init__(self, hidden_size):
        super(NNBase, self).__init__()
        self._hidden_size = hidden_size

    @property
    def output_size(self):
        return self._hidden_size


class MLPBase(NNBase):
    def __init__(self, num_inputs, hidden_size=64):
        super(MLPBase, self).__init__(hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class GNN(nn.Module):
    def __init__(self, transform, activation):
        super(GNN, self).__init__()

        self.transform = transform
        self.activation = activation

    def forward(self, adj, feat):
        seq = self.transform(feat)
        ret = t.matmul(seq, adj)
        return self.activation(ret)


class GNNBase(NNBase):
    def __init__(self, num_inputs, hidden_size=64):
        super(GNNBase, self).__init__(hidden_size)

        self.actor = nn.Sequential(
            GNN(init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh()),
            GNN(init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh()))

        self.critic = nn.Sequential(
            GNN(init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh()),
            GNN(init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh()))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, features, adj, rnn_hxs, masks):
        hidden_critic = self.critic(features, adj)
        hidden_actor = self.actor(features, adj)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        self.base = GNNBase(obs_shape['feat'].shape[0], **base_kwargs)

        self.dist = {}
        for k, space in action_space.items():
            if space.__class__.__name__ == "Discrete":
                self.dist[k] = [Categorical(self.base.output_size, space.n)]
            elif space.__class__.__name__ == "MultiDiscrete":
                self.dist[k] = [Categorical(self.base.output_size, v) for v in space.nvec]
            elif space.__class__.__name__ == "Box":
                num_outputs = action_space.shape[0]
                self.dist[k] = [DiagGaussian(self.base.output_size, num_outputs)]
            elif space.__class__.__name__ == "MultiBinary":
                num_outputs = action_space.shape[0]
                self.dist[k] = [Bernoulli(self.base.output_size, num_outputs)]
            else:
                raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, feat, task_id, masks, deterministic=False):
        # adj = inputs['adj']
        # feat = inputs['feat']
        # game_state = inputs['task_id']
        # placement = inputs['placement']
        # cards = inputs['cards']

        value, actor_features = self.base(feat, self.adj, masks)

        dists = [dist(actor_features) for dist in self.dist[task_id]]

        if deterministic:
            actions = [dist.mode() for dist in dists]
        else:
            actions = [dist.sample() for dist in dists]

        action_log_probs = [dist.log_probs(action) for dist, action in zip(dists, actions)]
        dist_entropy = [dist.entropy().mean() for dist in dists]

        return value, actions, action_log_probs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, feat, task_id, masks, actions):
        # adj = inputs['adj']
        # feat = inputs['feat']
        # game_state = inputs['task_id']
        # placement = inputs['placement']
        # cards = inputs['cards']
        value, actor_features = self.base(feat, self.adj, masks)
        dists = [dist(actor_features) for dist in self.dist[task_id]]

        action_log_probs = [dist.log_probs(action) for dist, action in zip(dists, actions)]
        dist_entropy = [dist.entropy().mean() for dist in dists]

        return value, action_log_probs, dist_entropy
