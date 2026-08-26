# PZ Risk

A [PettingZoo](https://pettingzoo.farama.org/) multi-agent reinforcement learning environment for the classic board game **Risk**. PZ Risk models a fully competitive, turn-based strategy game on a graph-based world map, supporting 2–6 agents.

> 📄 Associated paper: [NIPS 2022 — included in this repository as `NIPS2022.pdf`](./NIPS2022.pdf)

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Environments](#environments)
- [Game Mechanics](#game-mechanics)
  - [Game States](#game-states)
  - [Action Spaces](#action-spaces)
  - [Observation Space](#observation-space)
- [Maps](#maps)
- [Agents](#agents)
- [Wrappers](#wrappers)
- [Usage](#usage)
  - [Basic Usage](#basic-usage)
  - [Manual Play](#manual-play)
  - [Benchmarking](#benchmarking)
  - [Training with PPO](#training-with-ppo)
- [Project Structure](#project-structure)
- [Citation](#citation)

---

## Overview

PZ Risk implements the board game Risk as a multi-agent environment following the [PettingZoo AEC (Agent-Environment-Cycle)](https://pettingzoo.farama.org/api/aec/) API. The game is played on a graph where each node represents a territory and edges represent adjacency (attack routes). Players take turns reinforcing their territories, attacking neighbors, and fortifying positions — the last player with territories remaining wins.

Key features:
- **Multi-agent**: 2, 4, or 6 competitive agents
- **Graph-based board**: built with [NetworkX](https://networkx.org/)
- **PettingZoo & Gym compatible**: register and `gym.make(...)` supported
- **Multiple maps**: classic world map and smaller test configurations
- **Built-in agents**: Random, Greedy, Value-based, and Model-based
- **Observation wrappers**: vector and graph-based observations
- **Reward wrappers**: sparse and dense reward shaping

---

## Installation

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (the install script will fetch it if missing)

### One-command install

```bash
git clone https://github.com/mahi97/pz_risk.git
cd pz_risk
chmod +x install.sh
./install.sh          # environment, package, RL/evo extras
./install.sh --test   # same, then run pytest with coverage
```

`install.sh` creates a project virtualenv with **uv**, installs the package in editable mode, and pulls in every declared dependency: PettingZoo / Gymnasium, PyTorch, Stable-Baselines3, JAX, **evosax**, Flax, Optax, Gymnax, SuperSuit, and the pytest suite.

### Manual uv install

```bash
uv sync --all-extras --all-groups
uv run pytest
```

Activate the environment with `source .venv/bin/activate`, or prefix commands with `uv run`.

### Dependencies

| Package | Purpose |
|---|---|
| `gymnasium` | Spaces and single-agent env API |
| `pettingzoo` | Multi-agent AEC environment base |
| `networkx` | Graph representation of the board |
| `numpy` / `scipy` | Numerical operations |
| `matplotlib` | Rendering |
| `torch` | Existing PPO / DVN training stack |
| `stable-baselines3` | Additional RL algorithms (future work) |
| `jax` + `evosax` | Evolutionary strategies (future work) |
| `flax` / `optax` / `gymnax` | JAX RL stack (future work) |

---

## Environments

PZ Risk registers the following Gym environments:

| Environment ID | Players | Board |
|---|---|---|
| `Risk-Normal-2-v0` | 2 | World map |
| `Risk-Normal-4-v0` | 4 | World map |
| `Risk-Normal-6-v0` | 6 | World map |

```python
from pz_risk import make

env = make("Risk-Normal-6-v0")
```

The historical `import pz_risk.envs` still registers the named environments. Use `pz_risk.make(...)` rather than `gym.make(...)` — Risk is a PettingZoo AEC environment, not a single-agent Gymnasium `Env`.

---

## Game Mechanics

### Game States

Each turn progresses through an ordered sequence of states:

| State | Description |
|---|---|
| `StartTurn` | Begin a new turn; calculate reinforcement units |
| `Card` | Optionally trade in a set of 3 matching cards for bonus units |
| `Reinforce` | Place earned units onto owned territories (one at a time) |
| `Attack` | Optionally attack adjacent enemy territories |
| `Move` | Move surviving units after a successful attack |
| `Fortify` | Optionally move units between connected friendly territories |
| `EndTurn` | Game over (triggered when one player controls all territories) |

### Action Spaces

Action spaces are state-dependent:

| Game State | Action Space | Description |
|---|---|---|
| `Reinforce` | `Discrete(n_nodes)` | Index of territory to place 1 unit on |
| `Attack` | `MultiDiscrete([2, n_edges])` | `[skip, edge_index]` — 0 to attack, 1 to skip |
| `Move` | `Discrete(100)` | Number of units to move into captured territory |
| `Fortify` | `MultiDiscrete([2, n_nodes, n_nodes, 100])` | `[skip, src, dst, units]` |
| `Card` | `Discrete(2)` | 0 = skip, 1 = trade in best matching set |

### Observation Space

The raw observation returned by `observe(agent)` is the `Board` object, giving access to the full game graph and all state. Use the provided [wrappers](#wrappers) to convert this into a structured format suitable for learning algorithms.

### Cards

Cards are dealt to players when they successfully conquer a territory. Card types are `Infantry`, `Cavalry`, `Artillery`, and `Wild`. Trading in three matching cards (or one of each type) awards bonus placement units:

| Set | Bonus Units |
|---|---|
| 3× Infantry | 4 |
| 3× Cavalry | 6 |
| 3× Artillery | 8 |
| Wild set | 10 |

If a player holds 5 or more cards they are forced to trade.

---

## Maps

Maps are stored as JSON files under `pz_risk/maps/`. Each map defines nodes (territories), edges (adjacencies), group membership, and metadata.

| Map Name | Nodes | Description |
|---|---|---|
| `world` | 42 | Classic Risk world map |
| `8node` | 8 | Small 8-territory test map |
| `6node` | 6 | Small 6-territory test map |
| `4node` | 4 | Minimal 4-territory test map |

Custom maps can be added by creating a JSON file and calling `register_map(name, filepath)`.

---

## Agents

Built-in agents are located in `pz_risk/agents/`:

| Agent | Class | Description |
|---|---|---|
| **Random** | `RandomAgent` | Uniformly samples from valid actions |
| **Greedy** | `GreedyAgent` | Selects the action with the highest immediate advantage |
| **Value** | — | Uses a hand-crafted value function |
| **Model** | — | Model-based agent for planning |

All agents implement the `BaseAgent` interface with `reset()` and `act(state)` methods.

---

## Wrappers

PZ Risk provides several observation and reward wrappers in `pz_risk/wrappers/`:

| Wrapper | Description |
|---|---|
| `AssertInvalidActionsWrapper` | Raises an error if an invalid action is submitted |
| `VectorObservationWrapper` | Converts the board state to a flat numpy vector |
| `GraphObservationWrapper` | Converts the board state to a graph observation for GNN-based agents |
| `SparseRewardWrapper` | Provides +1 reward only when the game ends (win/lose) |
| `DenseRewardWrapper` | Provides per-step shaped rewards based on territory control |

Wrappers can be composed:

```python
from pz_risk import make
from pz_risk.wrappers import VectorObservationWrapper, SparseRewardWrapper

env = make("Risk-Normal-6-v0")
env = VectorObservationWrapper(env)
env = SparseRewardWrapper(env)
```

---

## Usage

### Basic Usage

```python
from pz_risk import make

env = make("Risk-Normal-6-v0")
env.reset()

for agent in env.agent_iter():
    obs, reward, terminated, truncated, info = env.last()
    if terminated or truncated:
        action = None
    else:
        action = env.unwrapped.sample()  # random valid action
    env.step(action)

env.close()
```

### Manual Play

Launch an interactive game where one agent is human-controlled (click-based) and the rest are random:

```bash
python manual.py --env Risk-Normal-6-v0 --num_agents 6 --num_manual 1
```

### Benchmarking

Measure environment throughput (reset time, rendering FPS, agent-step FPS):

```bash
python benchmark.py --env-name Risk-Normal-6-v0 --num_resets 200 --num_frames 5000
```

### Training with PPO

A PPO training script using the `GraphObservationWrapper` is included:

```bash
cd pz_risk
python train.py
```

A value-decomposition network (DVN) variant is also available via `train_v.py`.

---

## Project Structure

```
pz_risk/
├── agents/             # Built-in agent implementations
│   ├── base.py
│   ├── greedy.py
│   ├── model.py
│   ├── random.py
│   ├── sampling.py
│   └── value.py
├── core/               # Core game logic
│   ├── board.py        # Board, map loading, game step logic
│   ├── card.py         # Card types and scoring
│   ├── gamestate.py    # GameState enum
│   └── player.py       # Player state
├── envs/               # Gym-registered environment classes
│   └── normal.py
├── maps/               # Map definitions (JSON)
│   ├── world.json
│   ├── 4node.json
│   ├── 6node.json
│   └── 8node.json
├── training/           # PPO and DVN training infrastructure
├── wrappers/           # Observation and reward wrappers
├── risk_env.py         # Main RiskEnv (AECEnv) implementation
├── register.py         # Gym environment registration helper
├── utils.py            # Utility functions (dice rolling, etc.)
├── benchmark.py        # Throughput benchmarking script
├── manual.py           # Interactive manual play script
├── train.py            # PPO training entry point
└── train_v.py          # DVN training entry point
```

---

## Citation

If you use PZ Risk in your research, please cite:

```bibtex
@inproceedings{pzrisk2022,
  title     = {PZ Risk: A Multi-Agent Reinforcement Learning Environment for the Game of Risk},
  booktitle = {NeurIPS 2022},
  year      = {2022}
}
```

> See [`NIPS2022.pdf`](./NIPS2022.pdf) in this repository for the full paper.

