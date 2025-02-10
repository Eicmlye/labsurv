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
- [Tools Usage](#tools-usage)
  - [`Room`](#room)
  - [`SurveillanceRoom`](#surveillanceroom)

## Installation

```bash
# build virtual environment
python -m venv .

# install required packages
pip install -r requirements.txt
pip install -e .
```

> - [solution from CSDN](https://blog.csdn.net/Accele233/article/details/122416687)
>
> In case `github.com` is not available to you, here is a solution to `ssh: connect to host github.com port 22: Connection timed out`:
>
> 1. go to [ipaddress](https://www.ipaddress.com/website/www.github.com) and get latest ip for github.
> 2. on Windows OS, go to `C:/Windows/System32/drivers/etc/hosts`.
> 3. add `[IP] github.com` to this file, you may need to provide administor identity.
> 4. this ip may occasionally change, so update `hosts` if the problem raises again.

### Package Modification

- `pyntcloud`

  - For fewer warnings, change code in `Lib\site-packages\pyntcloud\io\ply.py` from

  ```python
    def describe_element(name, df):
        # ......
        else:
            for i in range(len(df.columns)):
                # get first letter of dtype to infer format
                f = property_formats[str(df.dtypes[i])[0]]  ### change this line !!!
                element.append('property ' + f + ' ' + df.columns.values[i])

        return element
  ```

  to

  ```python
    def describe_element(name, df):
        # ......
        else:
            for i in range(len(df.columns)):
                # get first letter of dtype to infer format
                f = property_formats[str(df.dtypes.iloc[i])[0]]  ### add `.iloc` !!!
                element.append('property ' + f + ' ' + df.columns.values[i])

        return element
  ```

## Quick Start

Make sure your current working directory is `labsurv/`, then

```bash
# train a new agent for cart pole
python tools/train.py --config configs/cart_pole/cart_pole.py

# test an existing agent for cart pole
# should change `load_from` in config file
python tools/test.py --config configs/cart_pole/cart_pole.py
```

## Code Style

This repository uses `black`, `isort` and `flake8` for code formatting.

To use shell on `Windows` OS, add your `Git` installation directory `git/bin/` to your system environmental variables. Then `sh` command is available on terminals.

To format changed files, run

```bash
sh infra/scripts/format.sh --git
```

And find more formatting options with `--help`.

## Project Structure

The project mainly uses `mmcv` registry module as the module administration system. An abstraction of the project structure is provided below.

![](https://github.com/Eicmlye/labsurv/blob/master/.readme/001_ProjectStructure.png)

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

## Tools Usage

### `SurveillanceRoom`

This class generates the `SurveillanceRoom` structure for the environment `BaseSurveillanceEnv`.

#### Quick start

```bash
python tools/demo/room/build_surveil_room.py
```

And find output in `output/surv_room/`.

You may also find generated samples in `tools/demo/room/samples/`.

#### Methods

The `SurveillanceRoom` class implements `add_block()` and `add_cam()` method as its main function. You may add `occupancy`, `install_permitted` and `must_monitor` blocks by `add_block()`, and add multiple sorts of cameras by `add_cam()`. The intrinsics of the cameras should be saved in `cfg_path` beforehand.

An example of the camera configuration is given in `configs\ocp\_base_\std_surveil.py`. You may simply add new types of cameras as value dict of `cam_intrinsics`, and it is recommended to choose from existed intrinsic params dict `clips`, `focals` and `resols`.
