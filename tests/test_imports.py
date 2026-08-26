"""Every public module must import, and third-party extras used later must be present."""

from __future__ import annotations

import importlib

import pytest

PACKAGE_MODULES = [
    "pz_risk",
    "pz_risk.register",
    "pz_risk.risk_env",
    "pz_risk.utils",
    "pz_risk.arena",
    "pz_risk.enjoy",
    "pz_risk.evaluate",
    "pz_risk.train",
    "pz_risk.train_v",
    "pz_risk.core",
    "pz_risk.core.board",
    "pz_risk.core.card",
    "pz_risk.core.gamestate",
    "pz_risk.core.player",
    "pz_risk.agents",
    "pz_risk.agents.base",
    "pz_risk.agents.greedy",
    "pz_risk.agents.model",
    "pz_risk.agents.random",
    "pz_risk.agents.sampling",
    "pz_risk.agents.value",
    "pz_risk.envs",
    "pz_risk.envs.normal",
    "pz_risk.wrappers",
    "pz_risk.wrappers.assert_invalid_actions",
    "pz_risk.wrappers.dense_reward",
    "pz_risk.wrappers.graph_observation",
    "pz_risk.wrappers.sparse_reward",
    "pz_risk.wrappers.vector_observation",
    "pz_risk.training",
    "pz_risk.training.arguments",
    "pz_risk.training.distributions",
    "pz_risk.training.dvn",
    "pz_risk.training.enjoy",
    "pz_risk.training.envs",
    "pz_risk.training.evaluation",
    "pz_risk.training.model",
    "pz_risk.training.ppo",
    "pz_risk.training.storage",
    "pz_risk.training.utils",
    "pz_risk.common",
    "pz_risk.common.logger",
    "pz_risk.common.preprocessing",
    "pz_risk.common.running_mean_std",
    "pz_risk.common.type_aliases",
    "pz_risk.common.utils",
    "pz_risk.common.envs",
    "pz_risk.common.sb2_compat",
    "pz_risk.common.sb2_compat.rmsprop_tf_like",
    "pz_risk.common.vec_env",
    "pz_risk.common.vec_env.base_vec_env",
    "pz_risk.common.vec_env.dummy_vec_env",
    "pz_risk.common.vec_env.subproc_vec_env",
    "pz_risk.common.vec_env.util",
    "pz_risk.common.vec_env.vec_normalize",
    "pz_risk.scripts",
    "pz_risk.scripts.manual",
    "pz_risk.scripts.benchmark",
    "pz_risk.maps",
]

THIRD_PARTY = [
    "gymnasium",
    "pettingzoo",
    "networkx",
    "numpy",
    "scipy",
    "matplotlib",
    "loguru",
    "torch",
    "stable_baselines3",
    "sb3_contrib",
    "evosax",
    "jax",
    "flax",
    "optax",
    "chex",
    "gymnax",
    "supersuit",
    "shimmy",
]


@pytest.mark.parametrize("name", PACKAGE_MODULES)
def test_package_module_imports(name):
    module = importlib.import_module(name)
    assert module is not None


@pytest.mark.parametrize("name", THIRD_PARTY)
def test_third_party_extra_imports(name):
    module = importlib.import_module(name)
    assert module is not None


def test_version_and_registry():
    import pz_risk
    from pz_risk import env_list, make

    assert pz_risk.__version__
    assert "Risk-Normal-2-v0" in env_list
    env = make("Risk-Normal-2-v0")
    env.reset()
    env.close()
