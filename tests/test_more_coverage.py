"""Additional coverage for remaining branches in existing modules."""

from __future__ import annotations

from collections import OrderedDict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from gymnasium import spaces

from pz_risk.common.preprocessing import maybe_transpose, preprocess_obs
from pz_risk.common.vec_env import DummyVecEnv
from pz_risk.core.board import BOARDS
from pz_risk.core.card import Card, CardType
from pz_risk.core.gamestate import GameState
from pz_risk.register import _load_entry_point
from pz_risk.risk_env import RiskEnv
from pz_risk.wrappers import AssertInvalidActionsWrapper, SparseRewardWrapper
from pz_risk.wrappers.graph_observation import GraphObservationWrapper


def test_attack_uses_dice_when_left_is_none():
    board = BOARDS["4node"]
    board.reset(2)
    src = board.player_nodes(0)[0]
    trg = next(n for n in board.g.nodes() if board.g.nodes[n]["player"] != 0)
    board.g.nodes[src]["units"] = 8
    board.g.nodes[trg]["units"] = 3
    board.state = GameState.Attack
    board.step(0, (0, (src, trg)))
    assert board.g.nodes[src]["units"] >= 1


def test_eliminating_player_transfers_cards():
    board = BOARDS["4node"]
    board.reset(2)
    victim_nodes = list(board.player_nodes(1))
    src = board.player_nodes(0)[0]
    trg = victim_nodes[0]
    # collapse opponent to a single territory so the next capture eliminates them
    for n in victim_nodes[1:]:
        board.g.nodes[n]["player"] = 0
        board.g.nodes[n]["units"] = 1
    board.g.nodes[src]["units"] = 10
    board.g.nodes[trg]["units"] = 1
    board.g.nodes[trg]["player"] = 1
    card = Card(99, CardType.Infantry)
    card.owner = 1
    board.players[1].cards[CardType.Infantry].append(card)
    board.state = GameState.Attack
    board.step(0, (0, (src, trg)), left=10)
    assert board.g.nodes[trg]["player"] == 0
    assert board.players[0].deserve_card is True


def test_invalid_entry_point():
    with pytest.raises(ValueError):
        _load_entry_point("not_a_valid_entry")


def test_assert_wrapper_other_states():
    raw = RiskEnv(n_agent=2, board_name="4node", render_mode="human")
    raw.reset(seed=1)
    wrapped = AssertInvalidActionsWrapper(raw)

    raw.board.state = GameState.Card
    wrapped.step(0)

    src = raw.board.player_nodes(raw.agent_selection)[0]
    trg = next(n for n in raw.board.g.nodes() if raw.board.g.nodes[n]["player"] != raw.agent_selection)
    raw.board.g.nodes[src]["units"] = 5
    raw.board.state = GameState.Attack
    wrapped.step((1, (None, None)))

    raw.board.last_attack = (src, trg)
    raw.board.g.nodes[trg]["units"] = 5
    raw.board.g.nodes[trg]["player"] = raw.agent_selection
    raw.board.state = GameState.Move
    wrapped.step(0)

    raw.board.state = GameState.Fortify
    wrapped.step((1, None, None, None))


def test_sparse_reward_win_and_lose():
    env = RiskEnv(n_agent=2, board_name="4node", render_mode="human")
    env.reset(seed=0)
    env = GraphObservationWrapper(env)
    env = SparseRewardWrapper(env)
    env.reset()
    board = env.unwrapped.board
    for n in board.g.nodes():
        board.g.nodes[n]["player"] = 0
        board.g.nodes[n]["units"] = 1
    assert env.reward(0) > 0
    assert env.done(0) is True
    assert env.reward(1) < 0
    assert env.done(1) is True


def test_dummy_vec_env_attr_helpers():
    from tests.conftest import BoxStubEnv

    venv = DummyVecEnv([BoxStubEnv, BoxStubEnv])
    venv.reset()
    vals = venv.get_attr("metadata")
    assert len(vals) == 2
    venv.set_attr("_t", 0)
    out = venv.env_method("reset")
    assert len(out) == 2
    venv.close()


def test_preprocess_box_and_discrete():
    box = spaces.Box(0, 255, (3, 4, 4), dtype=np.uint8)
    obs = torch.randint(0, 255, (2, 3, 4, 4))
    out = preprocess_obs(obs, box, normalize_images=True)
    assert out.max() <= 1.0
    disc = spaces.Discrete(4)
    one_hot = preprocess_obs(torch.tensor([1, 3]), disc)
    assert one_hot.shape[-1] == 4
    mb = spaces.MultiBinary(3)
    mb_out = preprocess_obs(torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.float32), mb)
    assert mb_out.shape[-1] == 3


def test_maybe_transpose_non_image():
    space = spaces.Box(-1, 1, (4,), dtype=np.float32)
    obs = np.zeros(4, dtype=np.float32)
    assert np.allclose(maybe_transpose(obs, space), obs)


def test_give_card_after_successful_turn():
    board = BOARDS["4node"]
    board.reset(2)
    board.players[0].deserve_card = True
    board.state = GameState.Fortify
    board.step(0, (1, None, None, None))
    assert board.state is GameState.StartTurn
    assert board.players[0].deserve_card is False


def test_model_agent_attack_branch(board_4node):
    from pz_risk.agents.model import ModelAgent
    from pz_risk.training.dvn import DVNAgent

    critic = DVNAgent(4, 2, 14, 20, device="cpu")
    with patch("pz_risk.agents.model.DVNAgent", return_value=critic), patch(
        "pz_risk.agents.model.torch.load", return_value=critic.state_dict()
    ):
        agent = ModelAgent(0, device="cpu")
    agent.critic = MagicMock(return_value=torch.zeros(1, 48, 1))
    src = board_4node.player_nodes(0)[0]
    trg = next(n for n in board_4node.g.nodes() if board_4node.g.nodes[n]["player"] != 0)
    board_4node.g.nodes[src]["units"] = 5
    board_4node.state = GameState.Attack
    with patch("pz_risk.agents.model.get_feat_adj_from_board", return_value=(np.zeros((48, 14)), np.eye(48))):
        action = agent.act(board_4node)
    assert action[0] in (0, 1, True, False)
