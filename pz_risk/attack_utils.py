# https://github.com/rmcarthur/gym-risk/blob/master/gym_risk/attack_utils.py

import logging
import random
import numpy as np

rng = np.random.default_rng()
sided_die = 6
attack_max = 3
defend_max = 2
attack_until_threshold = 2


def single_roll(attack: int, defend: int) -> (int, int):
    attack_roll = np.sort(rng.integers(1, sided_die + 1, min(attack_max, attack)))[::-1]
    defend_roll = np.sort(rng.integers(1, sided_die + 1, min(defend_max, defend)))[::-1]
    logging.info(f"Attack roll: {attack_roll}")
    logging.info(f"defend roll: {defend_roll}")
    max_loss = min(len(attack_roll), len(defend_roll))
    attack_wins = np.sum([i > j for i, j in zip(attack_roll, defend_roll)])
    attack_loss = max_loss - attack_wins
    defend_loss = attack_wins
    return attack_loss, defend_loss
