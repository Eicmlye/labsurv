[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

# labsurv

This repository is a reinforcement learning solution to the Optimal Camera Placement (OCP) problem.

## Contents

- [Installation](#installation)
- [Code Style](#code-style)
- [Project Structure](#project-structure)
  - [Runner](#runner)
  - [Agent](#agent)
  - [Environment](#environment)
  - [Replay Buffer](#replay-buffer)

## Installation

```bash
# build virtual environment
python -m venv .

# install required packages
pip install -r requirements.txt
pip install -e .
```

## Code Style

This repository uses `black`, `isort` and `flake8` for code formatting. To format changed files, run

```bash
sh infra/scripts/format.sh --git
```

And find more formatting options with `--help`.

## Project Structure

The project mainly uses `mmcv` registry module as the module administration system. An abstraction of the project structure is provided below.

![](.readme\001_ProjectStructure.png)

### Runner

The `Runner` class is the information processing center during the learning process. It deals with all the information from other components and checks if the episode truncates due to specific conditions.

### Agent

The `Agent` is the class to perform RL algorithms. It is recommended to inherit from `BaseAgent` class when creating a new `Agent`. 

An `Agent` must implement the `take_action()` and `update()` methods.

- `take_action()`: the `Agent` takes action according to the observation input and its exploration startegy.
- `update()`: the `Agent` updates its behaviour strategy (which is the target of optimization in RL) according to some experience sample(s), usually taken from the `ReplayBuffer`.

### Environment

The `Environment` is the class to model the extrinsic reward and terminating condition. It is designed to be a little bit different from the `gym.Envs` class, to better decouple the `Agent` and `Environment` and to adapt to dynamically changing `Environment`s. It is recommended to inherit from `BaseEnv` class when creating a new `Env`. 

An `Environment` must implement the `step()` and `reset()` methods.

- `step()`: Run one timestep of the environment's dynamics. The `Environment` takes in the current observation and action to perform a step. The observation is designed to be a sort of information spreading across the `Agent` and `Environment`, rather than a static state of `Environment`. 
- `reset()`: Reset a dynamimc environment to the initial state and returns an initial observation according to the initial state distribution. For static `Environment`s, there is no need to do state-initialization. `Environment` should specify its initial state distribution to generate `reset()` outputs.

### Replay Buffer

The `ReplayBuffer` is the class to store experience and to sample batches for `Agent` to update its strategy. A `BaseReplayBuffer` class is created for further uses. 