#!/usr/bin/env python3
"""Interactive Risk session with optional human-controlled agents."""

from __future__ import annotations

import argparse
import time

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseButton

from pz_risk import make

wait = True


def manual(agent, state):
    del agent, state
    global wait
    while wait:
        time.sleep(0.01)
    wait = True


def on_click(event):
    print(event)
    if event.button is MouseButton.LEFT:
        print("disconnecting callback")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        help="environment to load",
        default="Risk-Normal-6-v0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=-1,
    )
    parser.add_argument(
        "--num_agents",
        type=int,
        help="Number of Agents",
        default=6,
    )
    parser.add_argument(
        "--num_manual",
        default=1,
        help="Number of Manual Agents",
        type=int,
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    seed = None if args.seed is None or args.seed < 0 else args.seed
    env = make(args.env)
    env.reset(seed=seed)

    winner = -1
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    manual_agents = rng.choice(env.possible_agents, args.num_manual, replace=False)
    if len(manual_agents):
        print(manual_agents)
        plt.connect("button_press_event", on_click)

    for agent in env.agent_iter():
        obs, rew, terminated, truncated, info = env.last()
        if terminated or truncated:
            continue
        if agent in manual_agents:
            env.step(manual(agent, env.unwrapped.board.state))
        else:
            env.step(env.unwrapped.sample())
        if all(env.dones.values()):
            winner = agent
            break
        env.render()
    plt.show()
    logger.info(
        "Done in {} Turns and {} Moves. Winner is Player {}",
        env.unwrapped.num_turns,
        env.unwrapped.num_moves,
        winner,
    )


if __name__ == "__main__":
    main()
