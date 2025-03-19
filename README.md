[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

# labsurv

This repository is a reinforcement learning solution to the Optimal Camera Placement (OCP) problem of indoors scene.

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
- [Data Download](#data-download)

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

> If failed to install, try commenting the following lines in `requirements.txt`, and install pytorch & cudnn manually.
>
> ```plaintext
> # torch==2.3.1+cu121
> # torchaudio==2.3.1+cu121
> # torchvision==0.18.1+cu121
> ```

### Package Modification

- `pyntcloud`

  - For fewer warnings, change code in `Lib/site-packages/pyntcloud/io/ply.py` from

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
# build room pointcloud
python tools/demo/room/build_surveil_room.py

# train a new agent for cart pole
python tools/train.py --config configs/ocp/surveillance.py

# test an existing agent for cart pole
# should change `load_from` in config file
python tools/test.py --config configs/ocp/surveillance.py
```

## Code Style

This repository uses `black`, `isort` and `flake8` for code formatting.

To use shell on `Windows` OS, add your `Git` installation directory `git/bin/` to your system environmental variables. Then `sh` command is available on terminals.

To format changed files, run

```bash
sh infra/scripts/format.sh --git  # Windows
bash infra/scripts/format.sh --git  # Linux
```

And find more formatting options with `--help`.

## Project Structure

The project mainly uses `mmcv` registry module as the module administration system. An abstraction of the project structure is provided below.

### Runner

The `Runner` class is the information processing center during the learning process. It deals with all the information from other components and checks if the episode truncates due to specific conditions.

### Agent

The `Agent` is the class to perform RL algorithms. It is recommended to inherit from `BaseAgent` class when creating a new `Agent`.

An `Agent` must implement the `take_action()` and `update()` methods.

- `take_action()`: the `Agent` takes action according to the observation input and its exploration startegy.
- `update()`: the `Agent` updates its behaviour strategy (which is the target of optimization in RL) according to some experience sample(s), usually taken from the `ReplayBuffer`.

### Environment

The `Environment` models the extrinsic reward and terminating conditions. It is designed to be a little bit different from the `gym.Envs` class, to better decouple the `Agent` and `Environment` and to adapt to dynamically changing `Environment`s. It is recommended to inherit from `BaseEnv` class when creating a new `Env`.

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

An example of the camera configuration is given in `configs/ocp/_base_/std_surveil.py`. You may simply add new types of cameras as value dict of `cam_intrinsics`, and it is recommended to choose from existed intrinsic params dict `clips`, `focals` and `resols`.

### Visualization Tools

#### `plot.py`

Located at `labsurv/tools/ocp/`. This script plots reward and loss curves of the training process according to training logs.

- `--log`: if entered the path of the directory of the logs, the latest log will be analysed. If entered an exact file path, only this file will be analysed. After you run the traning, the log files are saved at `labsurv/output/ocp/AGENT_NAME/EXP_NAME/` and is named with a time stamp formatted as `yymmdd_hhmmss.log`.
- `--save`: path of the directory to save output images. If not specified, the directory of the log file will be used.
- `--step`: the step of the x-axis of the plots. Go to `labsurv/utils/plot/tick.py` and see docs of method `generate_absolute_ticks()` to get some examples on how x-ticks are generated.
- `--sma`: the window length of SMA operation on loss curves. If not specified, no SMA loss curves will be plotted.
- `--reward-sma`: the window length of SMA operation on reward curves. If not specified, no SMA reward curves will be plotted. Notice that this window length counts how many times evaluation is processed, but not how many episodes passed before an evaluation is completed.
- `--shrink`: takes out useful log lines and output a new log file. One may merge the shrank log to plot curves before and after resumation.
- `--drop-abnormal`: whether to drop abnormal values. The loss values are read from the log, and losses in the log files are printed with 6 decimal digits. Lower loss will drop to 1e-8 and the curve may be confusing. And this option drops these entries.

#### `reset_count_heatmap.py`

Located at `labsurv/tools/ocp/`. This script creates a point cloud heatmap of the visit counts.

- `--pkl`: the `.pkl` file of the visit counts. After you run the training, the `.pkl` file is saved at `labsurv/output/AGENT_NAME/EXP_NAME/envs/`.
- `--save`: path of the directory to save output pointcloud. If not specified, the directory of the `.pkl` file will be used.

## Data Download

We use the [Benchmark testbed on the Optimal Camera Placement Problem (OCP) and the Unicost Set Covering Problem (USCP)](https://www.mage.fst.uha.fr/brevilliers/ocp-uscp-benchmark/index.html0) to compare our method with existing methods. You may download the data `.txt` files [here](https://www.mage.fst.uha.fr/brevilliers/gecco-2021-ocp-uscp-competition/data.tar.xz), and the data structure is described [here](https://www.mage.fst.uha.fr/brevilliers/gecco-2021-ocp-uscp-competition/gecco_2021_ocp_uscp_competition.pdf).

### Modifications

#### `xx_specs.txt`

We suggest adding the size of the room to `xx_specs.txt` for easier data loading.

For `AC_specs.txt`,

```
AC_01 5 5 2 2.5 100 1920 1080 65 0.5 4
AC_02 10 10 2 2.5 100 1920 1080 65 0.5 4
AC_03 15 15 2 2.5 100 1920 1080 65 0.5 4
AC_04 20 20 2 2.5 100 1920 1080 65 0.5 4
AC_05 25 25 2 2.5 100 1920 1080 65 0.5 4
AC_06 30 30 2 2.5 100 1920 1080 65 0.5 4
AC_07 40 40 2 2.5 100 1920 1080 65 0.5 4
AC_08 50 50 2 2.5 100 1920 1080 65 0.5 4
AC_09 60 60 2 2.5 100 1920 1080 65 0.5 4
AC_10 5 5 2 2.5 500 1920 1080 65 0.5 4
AC_11 10 10 2 2.5 500 1920 1080 65 0.5 4
AC_12 15 15 2 2.5 500 1920 1080 65 0.5 4
AC_13 20 20 2 2.5 500 1920 1080 65 0.5 4
AC_14 25 25 2 2.5 500 1920 1080 65 0.5 4
AC_15 30 30 2 2.5 500 1920 1080 65 0.5 4
AC_16 40 40 2 2.5 500 1920 1080 65 0.5 4
AC_17 50 50 2 2.5 500 1920 1080 65 0.5 4
AC_18 60 60 2 2.5 500 1920 1080 65 0.5 4
AC_19 70 70 2 2.5 500 1920 1080 65 0.5 4
AC_20 80 80 2 2.5 500 1920 1080 65 0.5 4
AC_21 90 90 2 2.5 500 1920 1080 65 0.5 4
AC_22 100 100 2 2.5 500 1920 1080 65 0.5 4
AC_23 110 110 2 2.5 500 1920 1080 65 0.5 4
AC_24 120 120 2 2.5 500 1920 1080 65 0.5 4
AC_25 130 130 2 2.5 500 1920 1080 65 0.5 4
AC_26 140 140 2 2.5 500 1920 1080 65 0.5 4
AC_27 150 150 2 2.5 500 1920 1080 65 0.5 4
AC_28 160 160 2 2.5 500 1920 1080 65 0.5 4
AC_29 170 170 2 2.5 500 1920 1080 65 0.5 4
AC_30 180 180 2 2.5 500 1920 1080 65 0.5 4
AC_31 190 190 2 2.5 500 1920 1080 65 0.5 4
AC_32 200 200 2 2.5 500 1920 1080 65 0.5 4
```

### Check consistency

Because the benchmark provides covering information explicitly, and our algorithm computes covering relations by DDA algorithm, we have to check the consistency between these two covering set.

The operation below will generate the difference of the above coverings of the camera `[x, y, z, p, t]` in room `AC_01`.

```bash
python tools/ocp/data_check.py -n AC_01 -d <your_data_directory> --cam x y z p t
```

In the pointcloud, green points are covered by DDA, red points by benchmark, and orange by both.

### Train and test on benchmark

Change the room name variable `BENCHMARK_NAME` in `configs/ocp/_base_/params_benchmark.py`, and run

```bash
python tools/train.py --config configs/ocp/mappo_benchmark.py
```
