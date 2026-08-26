"""Training stack: DVN, PPO, storage, distributions, arguments, helpers."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box, Discrete, MultiBinary, MultiDiscrete
from torch import nn

from pz_risk.training.arguments import get_args
from pz_risk.training.distributions import Bernoulli, Categorical, DiagGaussian
from pz_risk.training.dvn import DVN, DVNAgent, GNN, ReplayMemory, Transition
from pz_risk.training.enjoy import parse_args as parse_enjoy_args
from pz_risk.training.envs import (
    MaskGoal,
    TransposeImage,
    TransposeObs,
    VecNormalize,
    VecPyTorch,
    make_env,
)
from pz_risk.training.evaluation import evaluate
from pz_risk.training.model import Flatten, GNN as PolicyGNN, GNNBase, MLPBase, Policy
from pz_risk.training.ppo import PPO
from pz_risk.training.storage import RolloutStorage, _flatten_helper
from pz_risk.training.utils import (
    AddBias,
    cleanup_log_dir,
    get_render_func,
    get_vec_normalize,
    init,
    init_,
    update_linear_schedule,
)


class _TinyActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def evaluate_actions(self, obs, task, masks, actions):
        values = self.linear(obs)
        logp = torch.zeros(obs.size(0), 1)
        entropy = torch.tensor(0.1)
        return values, logp, entropy, None

    def parameters(self, recurse=True):
        return super().parameters(recurse=recurse)


def test_replay_memory_and_dvn_train():
    mem = ReplayMemory(4)
    assert len(mem) == 0
    for i in range(5):
        mem.push(i, i, i, i, i, False)
    assert len(mem) == 4
    batch = mem.sample(2)
    assert len(batch) == 2

    feat_size = 6
    hidden = 8
    n_nodes, n_agents = 4, 2
    agent = DVNAgent(n_nodes, n_agents, feat_size, hidden, device="cpu")
    feat = torch.randn(1, n_nodes + n_agents, feat_size)
    adj = torch.eye(n_nodes + n_agents).unsqueeze(0)
    out = agent(feat, adj)
    assert out.shape[-1] == 1
    assert agent.train_start() is False

    for _ in range(agent.batch_size + 1):
        reward = torch.zeros(1, n_agents)
        done = torch.zeros(1, n_agents, dtype=torch.bool)
        agent.save_memory([feat, adj, reward, feat, adj, done])
    assert agent.train_start() is True
    loss = agent.train_()
    assert np.isfinite(loss)
    agent.num_train = agent.target_update - 1
    agent.train_()  # hits target update branch


def test_gnn_and_dvn_modules():
    gnn = GNN(nn.Linear(3, 4), nn.Tanh())
    feat = torch.randn(2, 5, 3)
    adj = torch.eye(5).unsqueeze(0).repeat(2, 1, 1)
    out = gnn(feat, adj)
    assert out.shape == (2, 5, 4)
    net = DVN(3, 4)
    values = net(feat, adj)
    assert values.shape == (2, 5, 1)


def test_distributions_forward():
    cat = Categorical(8, 4)
    dist = cat(torch.randn(3, 8))
    sample = dist.sample()
    assert sample.shape[0] == 3
    assert dist.log_probs(sample).shape[0] == 3
    assert dist.mode().shape[0] == 3

    gauss = DiagGaussian(8, 2)
    ndist = gauss(torch.randn(3, 8))
    nsample = ndist.sample()
    assert ndist.log_probs(nsample).shape[0] == 3
    assert ndist.entropy().shape[0] == 3
    assert torch.allclose(ndist.mode(), ndist.mean)

    bern = Bernoulli(8, 2)
    bdist = bern(torch.randn(3, 8))
    bsample = bdist.sample()
    assert bdist.log_probs(bsample).shape[0] == 3
    assert bdist.entropy().shape[0] == 3
    assert bdist.mode().shape[0] == 3


def test_mlp_gnn_flatten_policy():
    mlp = MLPBase(6, hidden_size=8)
    value, hidden, hx = mlp(torch.randn(2, 6), None, None)
    assert value.shape == (2, 1)
    assert hidden.shape[-1] == 8

    layer = PolicyGNN(nn.Linear(4, 4), nn.Tanh())
    feat = torch.randn(2, 4, 4)
    adj = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
    # Sequential-style GNN used by Policy expects (adj, feat)
    out = layer(adj, feat)
    assert out.shape[0] == 2

    flat = Flatten()
    assert flat(torch.randn(2, 3, 4)).shape == (2, 12)

    obs_spaces = {"feat": Box(0, 1, shape=(8,), dtype=np.float32)}
    action_spaces = {
        0: Discrete(3),
        1: MultiDiscrete([2, 4]),
    }
    policy = Policy(obs_spaces, action_spaces, base_kwargs={"hidden_size": 8})
    assert 0 in policy.dist and 1 in policy.dist


def test_rollout_storage_and_ppo(tmp_path):
    storage = RolloutStorage(
        num_steps=4,
        num_processes=2,
        obs_shape=(3,),
        action_space=Discrete(2),
        task_space=1,
    )
    storage.to("cpu")
    for step in range(4):
        storage.insert(
            obs=torch.randn(2, 3),
            task_id=torch.zeros(2, 1),
            actions=torch.zeros(2, 1),
            action_log_probs=torch.zeros(2),
            value_preds=torch.zeros(2, 1),
            rewards=torch.ones(2, 1),
            masks=torch.ones(2, 1),
            bad_masks=torch.ones(2, 1),
        )
    storage.compute_returns(torch.zeros(2, 1), gamma=0.99, use_proper_time_limits=True)
    storage.compute_returns(torch.zeros(2, 1), gamma=0.99, use_proper_time_limits=False)
    storage.after_update()
    advantages = storage.returns[:-1] - storage.value_preds[:-1]
    batches = list(storage.feed_forward_generator(advantages, num_mini_batch=2))
    assert batches
    empty = list(storage.feed_forward_generator(None, mini_batch_size=2))
    assert empty
    flat = _flatten_helper(2, 2, torch.zeros(2, 2, 3))
    assert flat.shape == (4, 3)

    actor = _TinyActor()
    ppo = PPO(
        actor,
        clip_param=0.2,
        ppo_epoch=1,
        num_mini_batch=2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        lr=1e-3,
        eps=1e-5,
        max_grad_norm=0.5,
        use_clipped_value_loss=True,
    )
    v, a, e = ppo.update(storage)
    assert all(np.isfinite(x) for x in (v, a, e))
    ppo.use_clipped_value_loss = False
    ppo.update(storage)


def test_training_utils(tmp_path):
    log_dir = tmp_path / "logs"
    cleanup_log_dir(str(log_dir))
    (log_dir / "foo.monitor.csv").write_text("x")
    cleanup_log_dir(str(log_dir))
    assert not (log_dir / "foo.monitor.csv").exists()

    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)
    update_linear_schedule(opt, epoch=5, total_num_epochs=10, initial_lr=0.1)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.05)

    layer = init(nn.Linear(3, 3), nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
    assert layer.bias.abs().sum() == 0
    layer2 = init_(nn.Linear(2, 2))
    assert layer2.weight.shape == (2, 2)

    bias = AddBias(torch.zeros(3))
    out = bias(torch.zeros(2, 3))
    assert out.shape == (2, 3)
    out4 = bias(torch.zeros(1, 3, 2, 2))
    assert out4.shape[1] == 3

    assert get_vec_normalize(object()) is None
    assert get_render_func(object()) is None

    class _Inner:
        def render(self):
            return "ok"

    class _V:
        envs = [_Inner()]

    assert callable(get_render_func(_V()))


def test_get_args_and_enjoy_parser():
    args = get_args(["--no-cuda", "--seed", "7", "--env-name", "Risk-Normal-2-v0"])
    assert args.seed == 7
    assert args.env_name == "Risk-Normal-2-v0"
    assert args.cuda is False
    enjoy = parse_enjoy_args(["--seed", "2", "--non-det"])
    assert enjoy.det is False


def test_make_env_factory_and_transpose():
    thunk = make_env("CartPole-v1", seed=0, rank=0)
    env = thunk()
    obs, info = env.reset()
    assert obs is not None
    env.close()

    class _Fake:
        observation_space = Box(0, 255, (4, 4, 3), dtype=np.uint8)
        _elapsed_steps = 1

        def observation(self, observation):
            return observation

    wrapped = TransposeImage(_as_env())
    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    assert wrapped.observation(frame).shape[0] == 3
    base = TransposeObs(_as_env())
    assert base is not None


class _SimpleEnv:
    def __init__(self):
        self.observation_space = Box(0, 255, (4, 5, 3), dtype=np.uint8)
        self.action_space = Discrete(2)
        self._elapsed_steps = 1

    def reset(self, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}

    def render(self):
        return np.zeros((4, 5, 3), dtype=np.uint8)

    def close(self):
        pass


def _as_env():
    import gymnasium as gym

    class E(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.observation_space = Box(0, 255, (4, 5, 3), dtype=np.uint8)
            self.action_space = Discrete(2)
            self._elapsed_steps = 1

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            return self.observation_space.sample(), {}

        def step(self, action):
            return self.observation_space.sample(), 0.0, False, False, {}

        def observation(self, observation):
            return observation

    return E()
