import argparse
import os
import os.path as osp
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from labsurv.utils.plot import generate_absolute_ticks
from labsurv.utils.string import WARN, to_filename
from matplotlib.axes import Axes


def parse_args():
    parser = argparse.ArgumentParser(description="Reward and loss figure painter.")

    parser.add_argument("--log", type=str, help="Path of the logger file.")
    parser.add_argument("--save", type=str, default=None, help="Path to save figures.")
    parser.add_argument(  # --single-fig
        "--single-fig", action="store_true", help="Whether plot as subfigures."
    )
    parser.add_argument("--step", type=int, default=20, help="The tick step number.")

    return parser.parse_args()


def get_latest_log(dir_name: str):
    assert osp.isdir(dir_name)

    filenames: List[str] = os.listdir(dir_name)
    latest_timestamp: int = 0
    latest_log: str = None
    for filename in filenames:
        if filename.endswith(".log"):
            cur_timestamp = filename.split(".")[0].split("_")
            try:
                cur_timestamp = int(cur_timestamp[0] + cur_timestamp[1])
            except TypeError:
                print(
                    WARN(
                        "Log files should be named in \"yymmdd_hhmmss.log\" format. "
                        f"{filename} is not a valid log file. "
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
    log_filename: str,
) -> Tuple[List[float], List[float], List[float], List[float], int]:
    # load y axis
    reward = []
    critic_loss = []
    actor_loss = []
    entropy = []
    eval_step = None
    found_episode_line = False
    with open(log_filename, "r") as f:
        for line in f:
            word_list = line.strip().split()

            if "====" in word_list and not found_episode_line:
                found_episode_line = True
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

            if "episode" in word_list:
                y_flag = word_list.index("episode")
            else:
                y_flag = -1

            if (
                y_flag > 0
                and y_flag < len(word_list) - 10
                and word_list[y_flag + 1] == "reward"
            ):
                critic_loss.append(float(word_list[y_flag + 6]))
                actor_loss.append(float(word_list[y_flag + 8]))
                entropy.append(float(word_list[y_flag + 10]))

            if "evaluation" in word_list:
                reward_flag = word_list.index("evaluation")
            else:
                reward_flag = -1

            if reward_flag > 0 and reward_flag < len(word_list) - 3:
                reward.append(float(word_list[reward_flag + 3]))

    return reward, critic_loss, actor_loss, entropy, eval_step


def plot_subfig(
    reward: List[float],
    critic_loss: List[float],
    actor_loss: List[float],
    entropy: List[float],
    eval_step: int,
    save_path: str,
    tick_step: int = 20,
):
    # figure settings
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    # plot graph
    valid_episode = min(len(critic_loss), len(actor_loss))
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
        entropy,
        line_style="-",
        color="y",
        title="entropy",
        tick_step=tick_step,
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

    # plt.show()
    fig.subplots_adjust(wspace=0.4, hspace=0.5)
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
    if log_if_needed and np.abs(max(y) / min(y)) > 100:
        log_scale = True
        y = np.array(np.log10(y)).tolist()
    ax.plot(x, y, line_style, color=color, label=title)
    ax.set_title(("log10 " if log_scale else "") + title)
    ax.set_xticks(generate_absolute_ticks(1, max(x), step=tick_step))


def plot_fig(
    reward: List[float],
    critic_loss: List[float],
    actor_loss: List[float],
    entropy: List[float],
    eval_step: int,
    save_dir_path: str,
    tick_step: int = 20,
):
    fig = plt.figure()
    for y, color, title in [
        [reward, "r", "reward"],
        [critic_loss, "g", "critic_loss"],
        [actor_loss, "b", "actor_loss"],
        [entropy, "y", "entropy"],
    ]:
        # figure settings
        fig.clear()

        log_scale = False
        if np.abs(max(y) / min(y)) > 1000:
            log_scale = True
            y = np.array(np.log10(y)).tolist()

        # plot graph
        x = [(i + 1) * (eval_step if title == "reward" else 1) for i in range(len(y))]
        plt.plot(x, y, "-", color=color)
        plt.xticks(
            generate_absolute_ticks(1, max(x), step=tick_step)
            if title == "reward"
            else generate_absolute_ticks(1, len(y), step=tick_step)
        )
        plt.title(title if not log_scale else ("log10 " + title))

        # plt.show()
        plt.savefig(osp.join(save_dir_path, title + ".png"), dpi=300, format="png")


def main():
    args = parse_args()

    if args.save is None:
        args.save = args.log

    log_filename = (
        get_latest_log(args.log) if not args.log.endswith(".log") else args.log
    )

    reward, critic_loss, actor_loss, entropy, eval_step = ocp_get_y_axis(log_filename)

    if args.single_fig:
        plot_subfig(
            reward,
            critic_loss,
            actor_loss,
            entropy,
            eval_step,
            to_filename(args.save, ".png", "reward_loss_fig"),
            tick_step=args.step,
        )
    else:
        assert osp.isdir(args.save)
        plot_fig(
            reward,
            critic_loss,
            actor_loss,
            entropy,
            eval_step,
            args.save,
            tick_step=args.step,
        )


if __name__ == "__main__":
    main()
