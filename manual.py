#!/usr/bin/env python3

import time
import argparse
import numpy as np
import gym
import pz_risk.envs
from pz_risk.wrappers import *

from matplotlib.backend_bases import MouseButton
from matplotlib import pyplot as plt
from loguru import logger

wait = True


def manual(agent, state):
    global wait
    while wait:
        time.sleep(0.01)
    wait = True


def on_click(event):
    print(event)
    if event.button is MouseButton.LEFT:
        print('disconnecting callback')


parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    help="gym environment to load",
    default='Risk-Normal-6-v0'
)
parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=-1
)
parser.add_argument(
    "--num_agents",
    type=int,
    help="Number of Agents",
    default=6
)
parser.add_argument(
    '--num_manual',
    default=1,
    help="Number of Manual Agents",
    type=int
)

args = parser.parse_args()

env = gym.make(args.env)
env.seed(args.seed)
env.reset()

winner = -1
manual_agents = np.random.choice(env.possible_agents, args.num_manual)
if len(manual_agents):
    print(manual_agents)
    plt.connect('button_press_event', on_click)

for agent in env.agent_iter():
    obs, rew, done, info = env.last()
    if done:
        continue
    if agent in manual_agents:
        env.step(manual(agent, env.unwrapped.board.state))
    else:
        env.step(env.unwrapped.sample())
    if all(env.dones.values()):
        winner = agent
        break
    env.render()
# e.render()
plt.show()
logger.info('Done in {} Turns and {} Moves. Winner is Player {}'
            .format(env.unwrapped.num_turns, env.unwrapped.num_moves, winner))
