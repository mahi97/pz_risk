"""CLI argument parsers and thin root-level entry modules."""

from __future__ import annotations

import importlib

from pz_risk.scripts.benchmark import parse_args as parse_benchmark
from pz_risk.scripts.manual import parse_args as parse_manual


def test_manual_parser_defaults():
    args = parse_manual([])
    assert args.env == "Risk-Normal-6-v0"
    assert args.num_agents == 6
    assert args.num_manual == 1
    args = parse_manual(["--env", "Risk-Normal-2-v0", "--seed", "4", "--num_manual", "0"])
    assert args.env == "Risk-Normal-2-v0"
    assert args.seed == 4
    assert args.num_manual == 0


def test_benchmark_parser():
    args = parse_benchmark([])
    assert args.env_name == "Risk-Normal-6-v0"
    args = parse_benchmark(["--env-name", "Risk-Normal-2-v0", "--num_resets", "3", "--num_frames", "10"])
    assert args.num_resets == 3
    assert args.num_frames == 10


def test_root_scripts_importable():
    assert importlib.import_module("manual").main is not None
    assert importlib.import_module("benchmark").main is not None
    arena = importlib.import_module("pz_risk.arena")
    assert callable(arena.main)
