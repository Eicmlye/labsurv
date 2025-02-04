import argparse
import os
import os.path as osp
from collections import deque
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
from labsurv.utils.plot import generate_absolute_ticks
from labsurv.utils.string import WARN, to_filename
from matplotlib.axes import Axes


def parse_args():
    parser = argparse.ArgumentParser(description="Reward and loss figure painter.")

    parser.add_argument("--log", type=str, help="Path of the logger file.")
    parser.add_argument("--save", type=str, default=None, help="Path to save figures.")
    parser.add_argument("--step", type=int, default=20, help="The tick step number.")
    parser.add_argument("--sma", type=int, default=1, help="The SMA window length for losses.")
    parser.add_argument(  # --reward-sma, -r
        "--reward-sma",
        "-r",
        type=int,
        default=1,
        help="The SMA window length for reward. Ignored if `--sma` == 1. ",
    )
    parser.add_argument(  # --shrink
        "--shrink",
        type=str,
        default=None,
        help="Whether to take out useful lines of the log.",
    )

    return parser.parse_args()


def get_latest_log(dir_name: str):
    assert osp.isdir(dir_name)

    filenames: List[str] = os.listdir(dir_name)
    latest_timestamp: int = 0
    latest_log: str = None
    for filename in filenames:
        if filename.endswith(".log") and not filename.endswith("shrink.log"):
            cur_timestamp = filename.split(".")[0].split("_")
            try:
                cur_timestamp = int(cur_timestamp[0] + cur_timestamp[1])
            except (TypeError, IndexError):
                print(
                    WARN(
                        "Log files should be named in \"yymmdd_hhmmss.log\" format. "
                        f"\"{filename}\" is not a valid log file. "
                    )
                )
                continue

            if latest_timestamp < cur_timestamp:
                latest_timestamp = cur_timestamp
                latest_log = filename

    if latest_log is None:
        raise ValueError(f"No valid log file found in {dir_name}.")

    return osp.join(dir_name, latest_log)


def ocp_get_y_axis(
    log_filename: str, shrink: Optional[str] = None
) -> Tuple[List[float], List[float], List[float], List[float], int]:
    # load y axis
    reward = []
    loss = []
    eval_step = None
    found_episode_line = False
    is_ac = False

    if shrink is not None:
        new_log = open(shrink, "w")

    with open(log_filename, "r") as f:
        for line in f:
            word_list = line.strip().split()

            if "====" in word_list and not found_episode_line:
                found_episode_line = True

                if shrink is not None:
                    new_log.write(line)

                if int(word_list[2]) != 1:
                    print(
                        WARN(
                            f"Current log file {log_filename} does not start from "
                            "episode 1. The x axis could be wrong. "
                        )
                    )

            if "Evaluating" in word_list and eval_step is None:
                if len(word_list) < 4:
                    raise ValueError(
                        f"Current log file {log_filename} is not a training log."
                    )
                eval_step = int(word_list[4])
                if shrink is not None:
                    new_log.write(line)

            if "episode" in word_list:
                y_flag = word_list.index("episode")
            else:
                y_flag = -1

            if (
                y_flag > 0
                and y_flag < len(word_list) - 5
                and word_list[y_flag + 1] == "reward"
            ):
                if shrink is not None:
                    new_log.write(line)
                    
                if "A" in word_list:
                    is_ac = True
                    actor_loss = (
                        float(word_list[y_flag + 8])
                        if word_list[y_flag + 7] == "A"
                        else float(word_list[y_flag + 6])
                    )
                    critic_loss = (
                        float(word_list[y_flag + 6])
                        if word_list[y_flag + 7] == "A"
                        else float(word_list[y_flag + 8])
                    )
                    entropy_loss = float(word_list[y_flag + 10])

                    if len(loss) == 0:
                        loss = [[actor_loss], [critic_loss], [entropy_loss]]
                    else:
                        loss[0].append(actor_loss)
                        loss[1].append(critic_loss)
                        loss[2].append(entropy_loss)
                else:
                    loss.append(float(word_list[y_flag + 5]))

            if "evaluation" in word_list:
                reward_flag = word_list.index("evaluation")
            else:
                reward_flag = -1

            if reward_flag > 0 and reward_flag < len(word_list) - 3:
                if shrink is not None:
                    new_log.write(line)
                    
                reward.append(float(word_list[reward_flag + 3]))

    if shrink is not None:
        new_log.close()

    return reward, loss, eval_step, is_ac


def plot_subfig(
    is_ac: bool,
    reward: List[float],
    loss: List[float],
    eval_step: int,
    save_path: str,
    tick_step: int = 20,
    sma: int = 1,
    reward_sma: int = 1,
):
    if is_ac:
        actor_loss, critic_loss, entropy_loss = loss
        _plot_ac_subfig(
            reward,
            actor_loss,
            critic_loss,
            entropy_loss,
            eval_step,
            save_path,
            tick_step,
            sma,
            reward_sma,
        )
    else:
        _plot_non_ac_subfig(reward, loss, eval_step, save_path, tick_step)


def _plot_ac_subfig(
    reward: List[float],
    actor_loss: List[float],
    critic_loss: List[float],
    entropy_loss: List[float],
    eval_step: int,
    save_path: str,
    tick_step: int = 20,
    sma: int = 1,
    reward_sma: int = 1,
):
    # figure settings
    if sma > 1:
        fig = plt.figure(figsize=(30, 10))
        ax1 = fig.add_subplot(2, 4, 1)
        ax2 = fig.add_subplot(2, 4, 2)
        ax3 = fig.add_subplot(2, 4, 3)
        ax4 = fig.add_subplot(2, 4, 4)
        if reward_sma > 1:
            ax5 = fig.add_subplot(2, 4, 5)
        ax6 = fig.add_subplot(2, 4, 6)
        ax7 = fig.add_subplot(2, 4, 7)
        ax8 = fig.add_subplot(2, 4, 8)
    else:
        fig = plt.figure(figsize=(20, 10))
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)

    # plot graph
    valid_episode = min(len(actor_loss), len(critic_loss), len(entropy_loss))
    x_reward = [(i + 1) * eval_step for i in range(len(reward))]
    x_loss = [(epi + 1) for epi in range(valid_episode)]

    _plot_subfig(
        ax1,
        x_reward,
        reward,
        line_style="-",
        color="r",
        title="reward",
        tick_step=tick_step,
    )
    _plot_subfig(
        ax2,
        x_loss,
        entropy_loss,
        line_style="-",
        color="y",
        title="entropy loss",
        tick_step=tick_step,
        log_if_needed=True,
    )
    _plot_subfig(
        ax3,
        x_loss,
        critic_loss,
        line_style="-",
        color="g",
        title="critic loss",
        tick_step=tick_step,
        log_if_needed=True,
    )
    _plot_subfig(
        ax4,
        x_loss,
        actor_loss,
        line_style="-",
        color="b",
        title="actor loss",
        tick_step=tick_step,
    )
    if sma > 1:
        if reward_sma > 1:
            _plot_subfig(
                ax5,
                x_reward,
                simple_moving_average(reward, window=reward_sma),
                line_style="-",
                color="r",
                title=f"SMA{reward_sma * eval_step} reward",
                tick_step=tick_step,
            )
        _plot_subfig(
            ax6,
            x_loss,
            simple_moving_average(entropy_loss, window=sma),
            line_style="-",
            color="y",
            title=f"SMA{sma} entropy loss",
            tick_step=tick_step,
            log_if_needed=True,
        )
        _plot_subfig(
            ax7,
            x_loss,
            simple_moving_average(critic_loss, window=sma),
            line_style="-",
            color="g",
            title=f"SMA{sma} critic loss",
            tick_step=tick_step,
            log_if_needed=True,
        )
        _plot_subfig(
            ax8,
            x_loss,
            simple_moving_average(actor_loss, window=sma),
            line_style="-",
            color="b",
            title=f"SMA{sma} actor loss",
            tick_step=tick_step,
        )

    # plt.show()
    fig.subplots_adjust(left=0.05, right=0.95, wspace=0.4, hspace=0.5)
    fig.savefig(save_path, dpi=300, format="png")


def _plot_non_ac_subfig(
    reward: List[float],
    loss: List[float],
    eval_step: int,
    save_path: str,
    tick_step: int = 20,
):
    # figure settings
    fig = plt.figure(figsize=(20, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    # plot graph
    valid_episode = len(loss)
    x_reward = [(i + 1) * eval_step for i in range(len(reward))]
    x_loss = [(epi + 1) for epi in range(valid_episode)]

    _plot_subfig(
        ax1,
        x_reward,
        reward,
        line_style="-",
        color="r",
        title="reward",
        tick_step=tick_step,
    )
    _plot_subfig(
        ax2,
        x_loss,
        loss,
        line_style="-",
        color="y",
        title="loss",
        tick_step=tick_step,
        log_if_needed=True,
    )

    # plt.show()
    fig.subplots_adjust(left=0.05, right=0.95, wspace=0.4, hspace=0.5)
    fig.savefig(save_path, dpi=300, format="png")


def _plot_subfig(
    ax: Axes,
    x: List[int],
    y: List[float],
    line_style: str = "-",
    color: str = "r",
    title: str = "loss",
    tick_step: int = 20,
    log_if_needed: bool = False,
):
    log_scale = False
    if log_if_needed and np.abs((max(y) + 1e-8) / (min(y) + 1e-8)) > 100:
        log_scale = True
        y = np.array(np.log10(np.array(y) + 1e-8)).tolist()
    ax.plot(x, y, line_style, color=color, label=title)
    ax.set_title(("log10 " if log_scale else "") + title)
    ax.set_xticks(generate_absolute_ticks(1, max(x), step=tick_step))


def simple_moving_average(vals: List[float], window: int = 10) -> List[float]:
    assert len(vals) >= window

    if window == 1:
        return vals.copy()

    sma_vals: List[float] = []
    cache: deque = deque(maxlen=window)

    for val in vals:
        cache.append(val)
        sma_vals.append(sum(cache) / len(cache))

    return sma_vals


def main():
    args = parse_args()

    if args.save is None:
        args.save = args.log if not args.log.endswith(".log") else osp.dirname(args.log)

    log_filename = (
        get_latest_log(args.log) if not args.log.endswith(".log") else args.log
    )

    reward, loss, eval_step, is_ac = ocp_get_y_axis(log_filename, args.shrink)

    plot_subfig(
        is_ac,
        reward,
        loss,
        eval_step,
        to_filename(args.save, ".png", "reward_loss_fig"),
        tick_step=args.step,
        sma=args.sma,
        reward_sma=args.reward_sma,
    )


if __name__ == "__main__":
    main()
