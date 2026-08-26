"""Utility helpers, sampling, value functions, and built-in agents."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from pz_risk.agents.base import BaseAgent
from pz_risk.agents.greedy import GreedyAgent
from pz_risk.agents.model import ModelAgent
from pz_risk.agents.random import RandomAgent
from pz_risk.agents.sampling import (
    SAMPLING,
    sample_attack,
    sample_card,
    sample_fortify,
    sample_move,
    sample_reinforce,
)
from pz_risk.agents.value import (
    get_attack_dist,
    get_chance,
    get_future,
    man_q_attack,
    man_q_deterministic,
    manual_advantage,
    manual_q,
    manual_value,
)
from pz_risk.core.board import BOARDS
from pz_risk.core.gamestate import GameState
from pz_risk.utils import flatten, get_feat_adj_from_board, single_roll, to_one_hot


def test_single_roll_losses_non_negative():
    rng = np.random.default_rng(0)
    for attack, defend in [(1, 1), (2, 1), (3, 2), (8, 5)]:
        a_loss, d_loss = single_roll(attack, defend)
        assert a_loss >= 0 and d_loss >= 0
        assert a_loss + d_loss <= min(3, attack, 2, defend) + 3


def test_to_one_hot_and_flatten():
    assert to_one_hot(2, 4) == [0, 0, 1, 0]
    assert list(flatten([1, [2, (3, 4)], "ab", 5])) == [1, 2, 3, 4, "ab", 5]
    assert list(flatten([])) == []


def test_get_feat_adj_from_board(board_4node):
    feats, adj = get_feat_adj_from_board(board_4node, player=0, n_agents=2, n_grps=1)
    n = board_4node.g.number_of_nodes() + 2
    assert len(feats) == n
    assert adj.shape == (n, n)
    assert np.allclose(np.diag(adj), 1.0)


def test_get_chance_base_cases():
    assert get_chance(3, 2, 100) == 0.0
    assert get_chance(-1, 1, 0) == 0.0
    assert get_chance(0, 2, -2) == 1.0
    assert get_chance(0, 2, 0) == 0.0
    assert get_chance(3, 0, 3) == 1.0
    assert get_chance(3, 0, 1) == 0.0
    p = get_chance(1, 1, 1)
    assert 0.0 < p < 1.0
    # cached
    assert get_chance(1, 1, 1) == p


def test_get_future_modes():
    dist = [(-1, 0.2), (0, 0.3), (2, 0.5)]
    assert get_future(dist, mode="safe") == -1
    assert get_future(dist, mode="risk", risk=0.25) == 0
    assert get_future(dist, mode="most") == 2
    expected_all = sum(d[0] * d[1] for d in dist)
    assert get_future(dist, mode="all") == pytest.approx(expected_all)
    # mode "two" splits on the probability sign (historical implementation)
    signed = [(-1, -0.2), (0, 0.3), (2, 0.5)]
    two = get_future(signed, mode="two", risk=0.1)
    assert isinstance(two, (int, float, np.floating))


def test_manual_value_and_q(board_4node):
    board = board_4node
    board.state = GameState.Reinforce
    node = board.player_nodes(0)[0]
    value = manual_value(board, 0)
    assert value > 0
    q = man_q_deterministic(board, 0, node)
    assert isinstance(q, (int, float, np.floating))
    adv = manual_advantage(board, 0, node)
    assert isinstance(adv, (int, float, np.floating))
    assert manual_q(board, 0, node) == q


def test_get_attack_dist_and_attack_q(board_4node):
    board = board_4node
    src = board.player_nodes(0)[0]
    trg = next(n for n in board.g.nodes() if board.g.nodes[n]["player"] != 0)
    board.g.nodes[src]["units"] = 4
    board.g.nodes[trg]["units"] = 2
    board.state = GameState.Attack
    assert get_attack_dist(board, (1, (None, None))) == []
    dist = get_attack_dist(board, (0, (src, trg)))
    assert dist
    assert all(len(item) == 2 for item in dist)
    q = man_q_attack(board, 0, (0, (src, trg)))
    assert isinstance(q, (int, float, np.floating))
    q_skip = man_q_attack(board, 0, (1, (None, None)))
    assert isinstance(q_skip, (int, float, np.floating))


def test_sampling_functions(board_4node):
    board = board_4node
    node = sample_reinforce(board, 0)
    assert node in board.player_nodes(0)
    assert sample_card(board, 0) in (0, 1)

    src = board.player_nodes(0)[0]
    trg = next(n for n in board.g.nodes() if n != src)
    board.g.nodes[src]["units"] = 5
    if board.g.nodes[trg]["player"] == 0:
        board.g.nodes[trg]["player"] = 1
    board.g.nodes[trg]["units"] = 1
    finished, edge = sample_attack(board, 0)
    assert finished in (0, 1, True, False)
    assert edge[0] in board.g.nodes()

    board.last_attack = (src, trg)
    board.g.nodes[trg]["units"] = 5
    move = sample_move(board, 0)
    assert move >= 0

    nodes = list(board.g.nodes())
    a, b = nodes[0], nodes[1]
    board.g.nodes[a]["player"] = 0
    board.g.nodes[b]["player"] = 0
    board.g.nodes[a]["units"] = 5
    board.g.nodes[b]["units"] = 5
    skip, s, t, units = sample_fortify(board, 0)
    assert skip in (0, 1, True, False)
    assert s != -1 and t != -1
    assert units >= 0
    assert GameState.Reinforce in SAMPLING
    assert SAMPLING[GameState.EndTurn](board, 0) is None


def test_base_and_random_agent(board_4node):
    base = BaseAgent()
    assert base.reset() is None
    assert base.act(None) is None

    agent = RandomAgent(0)
    agent.reset()
    board_4node.state = GameState.Reinforce
    action = agent.act(board_4node)
    assert action in board_4node.player_nodes(0)


def test_greedy_agent_reinforce(board_4node):
    agent = GreedyAgent(0)
    agent.reset()
    board_4node.state = GameState.Reinforce
    action = agent.act(board_4node)
    assert action in board_4node.player_nodes(0)


def test_model_agent_with_mocked_checkpoint(board_4node, tmp_path):
    feat_size = 14
    hidden = 20
    from pz_risk.training.dvn import DVNAgent

    critic = DVNAgent(4, 2, feat_size, hidden, device="cpu")
    ckpt = tmp_path / "8.pt"
    torch.save(critic.state_dict(), ckpt)

    with patch("pz_risk.agents.model.DVNAgent", return_value=critic), patch(
        "pz_risk.agents.model.torch.load", return_value=critic.state_dict()
    ), patch("pz_risk.agents.model.os.path.join", return_value=str(ckpt)):
        agent = ModelAgent(0, device="cpu")
    agent.reset()
    board_4node.state = GameState.Reinforce
    # ModelAgent hard-codes 48-node reshape; feed a tiny board through mocked critic
    agent.critic = MagicMock(return_value=torch.zeros(1, 48, 1))
    with patch("pz_risk.agents.model.get_feat_adj_from_board") as mocked:
        mocked.return_value = (np.zeros((48, 14)), np.eye(48))
        action = agent.act(board_4node)
    assert action in board_4node.player_nodes(0)
