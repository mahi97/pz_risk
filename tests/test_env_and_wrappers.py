"""RiskEnv, named environments, and observation / reward wrappers."""

from __future__ import annotations

import numpy as np
import pytest

from pz_risk import env_list, make
from pz_risk.core.gamestate import GameState
from pz_risk.envs.normal import Normal, NormalEnv2, NormalEnv4, NormalEnv6
from pz_risk.register import make as register_make
from pz_risk.risk_env import RiskEnv, env as make_risk_env
from pz_risk.wrappers import (
    AssertInvalidActionsWrapper,
    DenseRewardWrapper,
    GraphObservationWrapper,
    SparseRewardWrapper,
    VectorObservationWrapper,
)


def test_named_environments_registered():
    assert env_list == [
        "Risk-Normal-2-v0",
        "Risk-Normal-4-v0",
        "Risk-Normal-6-v0",
    ]
    two = make("Risk-Normal-2-v0")
    assert isinstance(two, NormalEnv2)
    assert two.n_agents == 2
    four = register_make("Risk-Normal-4-v0")
    assert isinstance(four, NormalEnv4)
    six = make("Risk-Normal-6-v0")
    assert isinstance(six, NormalEnv6)
    with pytest.raises(KeyError):
        make("Risk-Does-Not-Exist")


def test_normal_custom_board():
    env = Normal(n_agent=2, board_name="4node")
    env.reset(seed=1)
    assert env.n_nodes == 4
    assert env.observe(0) is env.board


def test_reset_is_seeded(raw_env):
    first = [raw_env.board.g.nodes[n]["units"] for n in raw_env.board.g.nodes()]
    raw_env.reset(seed=0)
    second = [raw_env.board.g.nodes[n]["units"] for n in raw_env.board.g.nodes()]
    raw_env.reset(seed=0)
    third = [raw_env.board.g.nodes[n]["units"] for n in raw_env.board.g.nodes()]
    assert second == third


def test_sample_and_step_loop(wrapped_env):
    env = wrapped_env
    steps = 0
    for agent in env.agent_iter(max_iter=80):
        obs, reward, terminated, truncated, info = env.last()
        assert "nodes" in info
        if terminated or truncated:
            env.step(None)
            continue
        action = env.unwrapped.sample()
        env.step(action)
        steps += 1
    assert steps > 0
    assert env.unwrapped.reward(0) == 0.0
    env.close()


def test_dones_property_and_done_helper(raw_env):
    assert set(raw_env.dones) == set(raw_env.possible_agents)
    assert all(v is False for v in raw_env.dones.values())
    assert raw_env.done(0) is False


def test_render_human_agg(raw_env):
    raw_env.render(mode="human")
    raw_env.render(mode="other")
    raw_env.close()


def test_graph_and_reward_wrappers():
    env = make_risk_env(n_agent=2, board_name="4node")
    env = GraphObservationWrapper(env)
    env = SparseRewardWrapper(env)
    env.reset()
    obs, reward, terminated, truncated, info = env.last()
    assert "feat" in obs and "adj" in obs and "task_id" in obs
    assert "rewards" in obs and "dones" in obs
    assert obs["adj"].shape[0] == env.unwrapped.n_nodes + env.unwrapped.n_agents
    action = env.unwrapped.sample()
    env.step(action)
    env.close()


def test_dense_and_vector_wrappers():
    env = make_risk_env(n_agent=2, board_name="4node")
    env = GraphObservationWrapper(env)
    env = DenseRewardWrapper(env)
    env.reset()
    obs, *_ = env.last()
    assert "rewards" in obs
    assert isinstance(obs["rewards"][0], (int, float, np.floating))
    env.close()

    env = make_risk_env(n_agent=2, board_name="4node")
    env = VectorObservationWrapper(env)
    env.reset()
    vec = env.observe(env.agent_selection)
    assert vec.shape[0] == env.unwrapped.n_nodes
    env.close()


def test_assert_invalid_actions_wrapper_accepts_valid():
    env = make_risk_env(n_agent=2, board_name="4node")
    env.reset()
    # factory already wraps with AssertInvalidActionsWrapper
    action = env.unwrapped.sample()
    env.step(action)
    env.close()


def test_assert_invalid_reinforce_raises():
    raw = RiskEnv(n_agent=2, board_name="4node")
    raw.reset(seed=0)
    wrapped = AssertInvalidActionsWrapper(raw)
    wrapped.reset()
    raw.board.state = GameState.Reinforce
    enemy = next(n for n in raw.board.g.nodes() if raw.board.g.nodes[n]["player"] != raw.agent_selection)
    with pytest.raises(AssertionError):
        wrapped.step(enemy)


def test_dead_agent_step_is_none(raw_env):
    raw_env.terminations[raw_env.agent_selection] = True
    raw_env.step(None)
    assert raw_env.agent_selection in raw_env.possible_agents


def test_end_turn_marks_everyone_done(raw_env):
    raw_env.board.state = GameState.EndTurn
    raw_env.step(raw_env.sample())
    assert all(raw_env.dones.values())


def test_wrapper_str():
    env = make_risk_env(n_agent=2, board_name="4node")
    env.reset()
    assert str(AssertInvalidActionsWrapper(env.unwrapped))
    assert str(VectorObservationWrapper(env))
    assert str(GraphObservationWrapper(env))
    assert str(SparseRewardWrapper(env))
    assert str(DenseRewardWrapper(env))
    env.close()
