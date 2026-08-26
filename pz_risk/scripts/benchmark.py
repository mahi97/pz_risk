#!/usr/bin/env python3
"""Throughput benchmark for Risk environments."""

from __future__ import annotations

import argparse
import time

from pettingzoo.utils import wrappers

from pz_risk import make
from pz_risk import wrappers as risk_wrappers


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-name",
        dest="env_name",
        help="environment to load",
        default="Risk-Normal-6-v0",
    )
    parser.add_argument("--num_resets", type=int, default=200)
    parser.add_argument("--num_frames", type=int, default=5000)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    env = make(args.env_name)

    t0 = time.time()
    for _ in range(args.num_resets):
        env.reset()
    t1 = time.time()
    reset_time = (1000 * (t1 - t0)) / args.num_resets

    t0 = time.time()
    for _ in range(args.num_frames):
        env.render()
    t1 = time.time()
    frames_per_sec = args.num_frames / (t1 - t0)

    env = make(args.env_name)
    env = wrappers.CaptureStdoutWrapper(env)
    env = risk_wrappers.AssertInvalidActionsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)

    env.reset()
    t0 = time.time()
    for agent in env.agent_iter(max_iter=args.num_frames):
        obs, rew, terminated, truncated, info = env.last()
        if terminated or truncated:
            env.step(None)
            continue
        env.step(env.unwrapped.sample())
    t1 = time.time()
    agent_view_fps = args.num_frames / (t1 - t0)

    print("Env reset time: {:.1f} ms".format(reset_time))
    print("Rendering FPS : {:.0f}".format(frames_per_sec))
    print("Agent view FPS: {:.0f}".format(agent_view_fps))


if __name__ == "__main__":
    main()
