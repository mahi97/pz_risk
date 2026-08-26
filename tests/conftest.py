"""Shared fixtures. Force a non-interactive matplotlib backend before any imports."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pytest

from gymnasium import spaces
import numpy as np

from pz_risk.core.board import BOARDS, Board
from pz_risk.risk_env import RiskEnv, env as make_risk_env


class BoxStubEnv:
    """Minimal env matching DummyVecEnv's historical observation_spaces API."""

    def __init__(self):
        space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
        self.observation_space = space
        self.observation_spaces = space
        self.action_space = spaces.Discrete(3)
        self.action_spaces = self.action_space
        self.metadata = {"render_modes": ["rgb_array"]}
        self._t = 0

    def reset(self):
        self._t = 0
        return self.observation_space.sample()

    def step(self, action):
        self._t += 1
        done = self._t >= 3
        return self.observation_space.sample(), 0.0, done, {}

    def seed(self, seed=None):
        return seed

    def render(self, mode="human"):
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def close(self):
        pass


@pytest.fixture
def board_4node() -> Board:
    board = BOARDS["4node"]
    board.reset(n_agent=2)
    return board


@pytest.fixture
def board_6node() -> Board:
    board = BOARDS["6node"]
    board.reset(n_agent=2)
    return board


@pytest.fixture
def raw_env() -> RiskEnv:
    e = RiskEnv(n_agent=2, board_name="4node")
    e.reset(seed=0)
    return e


@pytest.fixture
def wrapped_env():
    e = make_risk_env(n_agent=2, board_name="4node")
    e.reset(seed=0)
    return e
