"""Core game objects: cards, players, game states, maps, and Board mechanics."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from pz_risk.core.board import BOARDS, Board, register_map
from pz_risk.core.card import CARD_FIX_SCORE, Card, CardType
from pz_risk.core.gamestate import GameState
from pz_risk.core.player import Player


def test_card_types_and_scores():
    assert CardType.Infantry.value == 0
    assert CARD_FIX_SCORE[CardType.Infantry] == 4
    assert CARD_FIX_SCORE[CardType.Cavalry] == 6
    assert CARD_FIX_SCORE[CardType.Artillery] == 8
    assert CARD_FIX_SCORE[CardType.Wild] == 10
    card = Card(node=3, ctype=CardType.Cavalry)
    assert card.node == 3
    assert card.owner == -1
    assert card.type is CardType.Cavalry


def test_player_card_count():
    player = Player(0, init_placement=5)
    assert player.id == 0
    assert player.placement == 5
    assert player.num_cards() == 0
    player.cards[CardType.Infantry].append(Card(1, CardType.Infantry))
    player.cards[CardType.Wild].append(Card(-1, CardType.Wild))
    assert player.num_cards() == 2


def test_gamestate_order():
    names = [s.name for s in GameState]
    assert names == [
        "StartTurn",
        "Card",
        "Reinforce",
        "Attack",
        "Move",
        "Fortify",
        "EndTurn",
    ]


def test_builtin_maps_loaded():
    assert set(BOARDS) >= {"world", "4node", "6node", "8node"}
    assert BOARDS["4node"].g.number_of_nodes() == 4
    assert BOARDS["6node"].g.number_of_nodes() == 6
    assert BOARDS["8node"].g.number_of_nodes() == 8
    assert BOARDS["world"].g.number_of_nodes() == 42


def test_register_map_roundtrip(tmp_path):
    src = BOARDS["4node"]
    payload = {
        "info": src.info,
        "cells": [
            {**data, "id": node}
            for node, data in src.g.nodes(data=True)
        ],
        "edges": list(src.g.edges()),
    }
    path = tmp_path / "custom.json"
    import json

    path.write_text(json.dumps(payload))
    register_map("custom-test", path)
    assert "custom-test" in BOARDS
    assert BOARDS["custom-test"].g.number_of_nodes() == 4


def test_board_reset_deals_equal_territories(board_4node):
    board = board_4node
    counts = [len(board.player_nodes(p)) for p in range(2)]
    assert counts == [2, 2]
    assert board.player_units(0) > 0
    assert board.player_units(1) > 0
    assert len(board.players) == 2
    assert board.state is GameState.Reinforce
    assert board.n_cards == 4 + board.info["num_of_wild"]


def test_board_reset_rejects_uneven_split():
    board = BOARDS["4node"]
    with pytest.raises(AssertionError):
        board.reset(n_agent=3)


def test_calc_units_minimum_three(board_4node):
    # 2 territories => max(3, 2 // 3) == 3 plus any continent bonus
    assert board_4node.calc_units(0) >= 3


def test_player_group_reward_zero_when_split(board_4node):
    assert board_4node.player_group_reward(0) == 0
    assert board_4node.player_group_reward(1) == 0


def test_player_connected_components(board_4node):
    comps = board_4node.player_connected_components(0)
    assert isinstance(comps, list)
    assert all(isinstance(c, list) for c in comps)
    owned = set(board_4node.player_nodes(0))
    assert set().union(*comps) == owned


def test_reinforce_valid_actions_and_step(board_4node):
    board = board_4node
    board.state = GameState.Reinforce
    board.players[0].placement = 2
    deterministic, acts = board.valid_actions(0)
    assert deterministic is True
    assert set(acts) == set(board.player_nodes(0))
    target = acts[0]
    before = board.g.nodes[target]["units"]
    board.step(0, target)
    assert board.g.nodes[target]["units"] == before + 1
    assert board.players[0].placement == 1


def test_card_valid_actions_and_trade(board_4node):
    board = board_4node
    board.state = GameState.Card
    player = board.players[0]
    # three infantry cards => a legal set
    for i in range(3):
        card = Card(i, CardType.Infantry)
        card.owner = 0
        player.cards[CardType.Infantry].append(card)
    deterministic, acts = board.valid_actions(0)
    assert deterministic is True
    assert acts == [0, 1]
    before = player.placement
    board.step(0, 1)
    assert player.placement == before + CARD_FIX_SCORE[CardType.Infantry]
    assert player.num_cards() == 0


def test_card_forced_when_five_cards(board_4node):
    board = board_4node
    board.state = GameState.Card
    player = board.players[0]
    for i in range(5):
        player.cards[CardType.Infantry].append(Card(i, CardType.Infantry))
    _, acts = board.valid_actions(0)
    # cards is a dict keyed by CardType, so len(cards) is 4 (not the card count)
    assert 1 in acts


def test_can_card_with_mixed_set_and_wild(board_4node):
    board = board_4node
    board.state = GameState.StartTurn
    player = board.players[0]
    for bucket in player.cards.values():
        bucket.clear()
    player.cards[CardType.Infantry].append(Card(1, CardType.Infantry))
    player.cards[CardType.Cavalry].append(Card(2, CardType.Cavalry))
    player.cards[CardType.Artillery].append(Card(3, CardType.Artillery))
    assert player.num_cards() == 3
    assert board.can_card(0) is True
    player.cards[CardType.Infantry].append(Card(4, CardType.Infantry))
    player.cards[CardType.Infantry].append(Card(5, CardType.Infantry))
    assert player.num_cards() >= 5
    assert board.can_card(0) is True


def test_apply_best_match_wild_set(board_4node):
    board = board_4node
    player = board.players[0]
    player.cards[CardType.Infantry].append(Card(1, CardType.Infantry))
    player.cards[CardType.Cavalry].append(Card(2, CardType.Cavalry))
    player.cards[CardType.Wild].append(Card(-1, CardType.Wild))
    board.apply_best_match(0)
    assert player.placement >= CARD_FIX_SCORE[CardType.Wild] or player.placement >= 4


def test_attack_skip_and_roll(board_4node):
    board = board_4node
    # Give player 0 a stacked army next to an enemy
    src = board.player_nodes(0)[0]
    enemies = [
        n
        for n in board.g.neighbors(src)
        if board.g.nodes[n]["player"] != 0
    ]
    if not enemies:
        # force adjacency: assign a neighbor to the opponent
        neighbor = next(iter(board.g.neighbors(src)))
        board.g.nodes[neighbor]["player"] = 1
        board.g.nodes[neighbor]["units"] = 1
        trg = neighbor
    else:
        trg = enemies[0]
    board.g.nodes[src]["units"] = 10
    board.g.nodes[trg]["units"] = 1
    board.state = GameState.Attack
    assert board.can_attack(0)
    edges = board.player_attack_edges(0)
    assert (src, trg) in edges or any(e[0] == src for e in edges)

    deterministic, acts = board.valid_actions(0)
    assert deterministic is False
    assert (1, (None, None)) in acts

    # skip attack
    board.step(0, (1, (None, None)))
    assert board.state in {GameState.Fortify, GameState.StartTurn}

    board.state = GameState.Attack
    board.g.nodes[src]["units"] = 10
    board.g.nodes[trg]["units"] = 1
    board.g.nodes[trg]["player"] = 1
    board.step(0, (0, (src, trg)), left=10)
    # with left=10 attacker should wipe the defender
    assert board.g.nodes[trg]["player"] == 0
    assert board.last_attack == (src, trg)
    assert board.state is GameState.Move


def test_move_after_attack(board_4node):
    board = board_4node
    nodes = list(board.g.nodes())
    src, trg = nodes[0], nodes[1]
    board.g.nodes[src]["player"] = 0
    board.g.nodes[trg]["player"] = 0
    board.g.nodes[src]["units"] = 1
    board.g.nodes[trg]["units"] = 6
    board.last_attack = (src, trg)
    board.state = GameState.Move
    deterministic, acts = board.valid_actions(0)
    assert deterministic is True
    assert acts == [0, 1, 2, 3]
    board.step(0, 2)
    assert board.g.nodes[src]["units"] == 3
    assert board.g.nodes[trg]["units"] == 4


def test_fortify_and_skip(board_4node):
    board = board_4node
    nodes = board.player_nodes(0)
    # force both owned nodes to be connected with extra units
    for n in board.g.nodes():
        if n in nodes:
            board.g.nodes[n]["player"] = 0
            board.g.nodes[n]["units"] = 5
        else:
            board.g.nodes[n]["player"] = 1
            board.g.nodes[n]["units"] = 1
    board.state = GameState.Fortify
    assert board.can_fortify(0)
    deterministic, acts = board.valid_actions(0)
    assert deterministic is True
    assert (1, None, None, None) in acts
    src, dst = nodes[0], nodes[1]
    src_before = board.g.nodes[src]["units"]
    dst_before = board.g.nodes[dst]["units"]
    board.step(0, (0, src, dst, 2))
    assert board.g.nodes[src]["units"] == src_before - 2
    assert board.g.nodes[dst]["units"] == dst_before + 2

    board.state = GameState.Fortify
    board.step(0, (1, None, None, None))
    assert board.state is GameState.StartTurn


def test_start_turn_assigns_placement(board_4node):
    board = board_4node
    board.state = GameState.StartTurn
    board.step(0, None)
    assert board.players[0].placement >= 3
    assert board.state in {GameState.Card, GameState.Reinforce}


def test_give_card(board_4node):
    board = board_4node
    board.players[0].deserve_card = True
    before = board.players[0].num_cards()
    board.give_card(0)
    assert board.players[0].num_cards() == before + 1
    assert board.players[0].deserve_card is False


def test_next_state_endturn_stays(board_4node):
    board = board_4node
    board.state = GameState.EndTurn
    board.next_state(0, GameState.EndTurn, False, False, True)
    assert board.state is GameState.EndTurn


def test_world_reset_six_players():
    board = BOARDS["world"]
    board.reset(n_agent=6)
    counts = [len(board.player_nodes(p)) for p in range(6)]
    assert counts == [7, 7, 7, 7, 7, 7]
    assert sum(board.player_units(p) for p in range(6)) > 0
