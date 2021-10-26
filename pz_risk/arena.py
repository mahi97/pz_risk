from risk_env import env
from agents import GreedyAgent, RandomAgent, ModelAgent
from loguru import logger
import matplotlib.pyplot as plt

e = env()
e.reset()

players = [GreedyAgent(i) for i in range(2)]
players += [GreedyAgent(2 + i) for i in range(2)]
players += [RandomAgent(4 + i) for i in range(2)]
winner = -1
for agent in e.agent_iter():
    obs, rew, done, info = e.last()
    if done:
        continue
    e.step(players[agent].act(obs))
    for a in e.possible_agents:
        e.unwrapped.land_hist[a].append(len(e.unwrapped.board.player_nodes(a)))
        e.unwrapped.unit_hist[a].append(e.unwrapped.board.player_units(a))
        e.unwrapped.place_hist[a].append(e.unwrapped.board.calc_units(a))
    if all(e.dones.values()):
        winner = agent
        break
    e.render()
e.render()
plt.show()
logger.info('Done in {} Turns and {} Moves. Winner is Player {}'
            .format(e.unwrapped.num_turns, e.unwrapped.num_moves, winner))
