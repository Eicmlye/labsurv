import argparse
import os
import os.path as osp
from collections import deque
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from labsurv.utils.plot import generate_absolute_ticks
from labsurv.utils.string import INDENT, WARN, to_filename
from matplotlib.axes import Axes


def parse_args():
    parser = argparse.ArgumentParser(description="Reward and loss figure painter.")

    parser.add_argument("--log", type=str, help="Path of the logger file.")
    parser.add_argument("--save", type=str, default=None, help="Path to save figures.")
    parser.add_argument("--step", type=int, default=20, help="The tick step number.")
    parser.add_argument(  # --sma
        "--sma", type=int, default=1, help="The SMA window length for losses."
    )
    parser.add_argument(  # --reward-sma, -r
        "--reward-sma",
        "-r",
        type=int,
        default=1,
        help="The SMA window length for reward. Ignored if `--sma` == 1. ",
    )
    parser.add_argument(  # --shrink
        "--shrink",
        action="store_true",
        help="Whether to take out useful lines of the log.",
    )
    parser.add_argument(  # --plot, -p
        "--plot",
        "-p",
        action="store_true",
        help="Whether to plot loss curve.",
    )
    parser.add_argument(  # --cov, -c
        "--cov",
        "-c",
        action="store_true",
        help="Whether to plot final coverage.",
    )
    parser.add_argument(  # --eta, -e
        "--eta",
        "-e",
        type=int,
        default=0,
        help="Whether to print eta on the terminal.",
    )

    return parser.parse_args()


def get_latest_log(dir_name: str):
    assert osp.isdir(dir_name)

    filenames: List[str] = os.listdir(dir_name)
    latest_timestamp: int = 0
    latest_log: str = None
    for filename in filenames:
        if (
            filename.endswith(".log")
            and not filename.endswith("shrink.log")
            and not filename.endswith("cov.log")
        ):
            cur_timestamp = filename.split(".")[0].split("_")
            try:
                cur_timestamp = int(cur_timestamp[0] + cur_timestamp[1])
            except (TypeError, IndexError, ValueError):
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


def get_eta(log_filename: str, etas: int = 0):
    assert etas > 0

    time_stamp = deque(maxlen=etas)
    latest_eta = deque(maxlen=etas)
    latest_elapsed = deque(maxlen=etas)

    with open(log_filename, "r") as f:
        for line in f:
            word_list = line.strip().split()

            if "eta:" in word_list:
                time_stamp.append(" ".join(word_list[:2]))

                eta_index = word_list.index("eta:")
                eta_str_len = 3 if word_list[eta_index + 2].startswith("day") else 1
                latest_eta.append(
                    " ".join(word_list[eta_index : eta_index + eta_str_len + 1])
                )

                elapsed_index = word_list.index("elapsed:")
                elapsed_str_len = (
                    3 if word_list[elapsed_index + 2].startswith("day") else 1
                )
                latest_elapsed.append(
                    " ".join(
                        word_list[elapsed_index : elapsed_index + elapsed_str_len + 1]
                    )
                )

    for time, eta, elapsed in zip(time_stamp, latest_eta, latest_elapsed):
        print(time + INDENT + eta + " | " + elapsed)


def ocp_get_cov(
    log_filename: str, shrink: Optional[str] = None
) -> Tuple[List[float], List[float], int]:
    train_covs = []
    eval_covs = []
    eval_step = None
    found_episode_line = False
    start_episode: int = 1

    pred_train_cov = None
    pred_eval_cov = None

    shrinked_log: bool = log_filename.endswith("cov.log")

    if shrink is not None:
        new_log = open(shrink, "w")

    with open(log_filename, "r") as f:
        for line in f:
            word_list = line.strip().split()

            if "====" in word_list and not found_episode_line:
                found_episode_line = True

                if shrink is not None:
                    new_log.write(line)

                start_episode = int(word_list[2])

            if "Evaluating" in word_list and eval_step is None:
                if len(word_list) < 4:
                    raise ValueError(
                        f"Current log file {log_filename} is not a training log."
                    )
                eval_step = int(word_list[4]) - start_episode + 1
                if shrink is not None:
                    new_log.write(line)

            if (
                not shrinked_log and len(word_list) >= 5 and word_list[4].endswith("%")
            ) or (
                shrinked_log and word_list[0] == "eval" and word_list[1].endswith("%")
            ):
                pred_eval_cov = eval(word_list[1 if shrinked_log else 4][:-1]) / 100
            elif (
                not shrinked_log and len(word_list) >= 7 and word_list[6].endswith("%")
            ) or (
                shrinked_log and word_list[0] == "train" and word_list[1].endswith("%")
            ):
                pred_train_cov = eval(word_list[1 if shrinked_log else 6][:-1]) / 100

            if "evaluation" in word_list:
                eval_covs.append(pred_eval_cov)
                if shrink is not None:
                    new_log.write(f"eval {100 * pred_eval_cov:.4f}%\n" + line)
            elif "episode reward" in line:
                train_covs.append(pred_train_cov)
                if shrink is not None:
                    new_log.write(f"train {100 * pred_train_cov:.4f}%\n" + line)

    if shrink is not None:
        new_log.close()

    return train_covs, eval_covs, eval_step


def plot_cov(
    train_covs: List[float],
    eval_covs: List[float],
    eval_step: int,
    save_path: str,
    tick_step: int = 20,
    sma: int = 1,
    eval_sma: int = 1,
):
    if sma > 1:
        fig = plt.figure(figsize=(25, 10))
        ax_train = fig.add_subplot(1, 2, 1)  # train_cov
        ax_eval = fig.add_subplot(1, 2, 2)  # eval_cov
    else:
        raise NotImplementedError()

    x_train = [(epi + 1) for epi in range(len(train_covs))]
    x_eval = [(i + 1) * eval_step for i in range(len(eval_covs))]

    _plot_subfig(
        ax_train,
        x_train,
        train_covs,
        line_style="-",
        color="r",
        title="train coverage",
        tick_step=tick_step,
        label="actual",
    )
    _plot_subfig(
        ax_eval,
        x_eval,
        eval_covs,
        line_style="-",
        color="r",
        title="eval coverage",
        tick_step=tick_step,
        label="actual",
    )

    if sma > 1:
        _plot_subfig(
            ax_train,
            x_train,
            train_covs,
            line_style="-",
            color="black",
            tick_step=tick_step,
            label=f"SMA{sma}",
            sma=sma,
        )
        _plot_subfig(
            ax_eval,
            x_eval,
            eval_covs,
            line_style="-",
            color="black",
            tick_step=tick_step,
            label=f"SMA{eval_sma * eval_step}",
            sma=eval_sma,
            sma_tick_extend_factor=eval_step,
        )

    # plt.show()
    fig.subplots_adjust(left=0.05, right=0.95, wspace=0.4, hspace=0.5)
    fig.savefig(save_path, dpi=300, format="png")


def ocp_get_y_axis(
    log_filename: str, shrink: Optional[str] = None
) -> Tuple[
    List[float], List[float], List[List[float]], int, bool, Optional[List[List[float]]]
]:
    # load y axis
    train_reward = []
    eval_reward = []
    loss: List[Optional[List[float]]] = []
    disc_output = None
    eval_step = None
    found_episode_line = False
    start_episode: int = 1
    is_ac: bool = False

    if shrink is not None:
        new_log = open(shrink, "w")

    with open(log_filename, "r") as f:
        for line in f:
            word_list = line.strip().split()

            if "====" in word_list and not found_episode_line:
                found_episode_line = True

                if shrink is not None:
                    new_log.write(line)

                start_episode = int(word_list[2])

            if "Evaluating" in word_list and eval_step is None:
                if len(word_list) < 4:
                    raise ValueError(
                        f"Current log file {log_filename} is not a training log."
                    )
                eval_step = int(word_list[4]) - start_episode + 1
                if shrink is not None:
                    new_log.write(line)

            # loss line
            if "episode" in word_list:
                y_flag = word_list.index("episode")
            else:
                y_flag = -1

            if (
                y_flag >= 0
                and y_flag < len(word_list) - 5
                and word_list[y_flag + 1] == "reward"
            ):
                if shrink is not None:
                    new_log.write(line)

                train_reward.append(float(word_list[y_flag + 2]))

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
                    if "D" in word_list:
                        disc_loss = float(word_list[y_flag + 12])

                    if len(loss) == 0:
                        loss = [[actor_loss], [critic_loss], [entropy_loss], None]
                        if "D" in word_list:
                            loss[3] = [disc_loss]
                    else:
                        loss[0].append(actor_loss)
                        loss[1].append(critic_loss)
                        loss[2].append(entropy_loss)
                        if "D" in word_list:
                            loss[3].append(disc_loss)
                else:
                    loss.append(float(word_list[y_flag + 5]))

            # reward line
            if "evaluation" in word_list:
                reward_flag = word_list.index("evaluation")
            else:
                reward_flag = -1

            if reward_flag >= 0 and reward_flag < len(word_list) - 3:
                if shrink is not None:
                    new_log.write(line)

                eval_reward.append(float(word_list[reward_flag + 3]))

            # discriminator performance line
            if "Discriminator" in word_list:
                disc_flag = word_list.index("Discriminator")
            else:
                disc_flag = -1

            if disc_flag >= 0:
                if shrink is not None:
                    new_log.write(line)

                acc = float(word_list[disc_flag + 3])
                prec = float(word_list[disc_flag + 7])
                recall = float(word_list[disc_flag + 11])
                if disc_output is None:
                    disc_output = [[acc], [prec], [recall]]
                else:
                    disc_output[0].append(acc)
                    disc_output[1].append(prec)
                    disc_output[2].append(recall)

    if shrink is not None:
        new_log.close()

    if np.all(np.array(loss[3]) == 0):
        loss[3] = None

    return train_reward, eval_reward, loss, eval_step, is_ac, disc_output


def plot_subfig(
    is_ac: bool,
    train_reward: List[float],
    eval_reward: List[float],
    loss: List[float],
    eval_step: int,
    save_path: str,
    tick_step: int = 20,
    sma: int = 1,
    reward_sma: int = 1,
    disc_output: Optional[List[List[float]]] = None,
):
    if is_ac:
        actor_loss, critic_loss, entropy_loss, disc_loss = loss
        disc_acc, disc_prec, disc_recall = (
            disc_output if disc_output is not None else (None, None, None)
        )
        _plot_ac_subfig(
            train_reward,
            eval_reward,
            actor_loss,
            critic_loss,
            entropy_loss,
            eval_step,
            save_path,
            tick_step=tick_step,
            sma=sma,
            reward_sma=reward_sma,
            disc_loss=disc_loss,
            disc_acc=disc_acc,
            disc_prec=disc_prec,
            disc_recall=disc_recall,
        )
    else:
        _plot_non_ac_subfig(eval_reward, loss, eval_step, save_path, tick_step)


def _plot_ac_subfig(
    train_reward: List[float],
    eval_reward: List[float],
    actor_loss: List[float],
    critic_loss: List[float],
    entropy_loss: List[float],
    eval_step: int,
    save_path: str,
    tick_step: int = 20,
    sma: int = 1,
    reward_sma: int = 1,
    disc_loss: Optional[List[float]] = None,
    disc_acc: Optional[List[float]] = None,
    disc_prec: Optional[List[float]] = None,
    disc_recall: Optional[List[float]] = None,
):
    # figure settings
    if sma > 1:
        fig = plt.figure(figsize=(40, 20))
        ax_train = fig.add_subplot(3, 5, 1)  # train_reward
        ax_train_sma = fig.add_subplot(3, 5, 2)  # train_reward sma
        ax_eval = fig.add_subplot(3, 5, 3)  # eval_reward
        if reward_sma > 1:
            ax_eval_sma = fig.add_subplot(3, 5, 4)  # eval_reward sma
        ax_ent = fig.add_subplot(3, 5, 6)  # entropy loss
        ax_ent_sma = fig.add_subplot(3, 5, 7)  # entropy loss sma
        ax_critic = fig.add_subplot(3, 5, 8)  # critic loss
        ax_critic_sma = fig.add_subplot(3, 5, 9)  # critic loss sma
        ax_actor = fig.add_subplot(3, 5, 11)  # actor loss
        ax_actor_sma = fig.add_subplot(3, 5, 12)  # actor loss sma
        ax_disc = fig.add_subplot(3, 5, 13)  # disc loss
        ax_disc_sma = fig.add_subplot(3, 5, 14)  # disc loss sma

        ax_disc_acc = fig.add_subplot(3, 5, 5)  # disc accuracy
        ax_disc_prec = fig.add_subplot(3, 5, 10)  # disc precision
        ax_disc_recall = fig.add_subplot(3, 5, 15)  # disc recall
    else:
        raise NotImplementedError()

    # plot graph
    loss_valid_episode = min(len(actor_loss), len(critic_loss), len(entropy_loss))
    if disc_loss is not None:
        loss_valid_episode = min(loss_valid_episode, len(disc_loss))
        disc_valid_episode = min(len(disc_acc), len(disc_prec), len(disc_recall))

    x_reward = [(i + 1) * eval_step for i in range(len(eval_reward))]
    x_loss = [(epi + 1) for epi in range(loss_valid_episode)]
    if disc_loss is not None:
        x_disc = [(epi + 1) for epi in range(disc_valid_episode)]

    _plot_subfig(
        ax_train,
        x_loss,
        train_reward,
        line_style="-",
        color="r",
        title="train reward",
        tick_step=tick_step,
    )
    _plot_subfig(
        ax_eval,
        x_reward,
        eval_reward,
        line_style="-",
        color="r",
        title="eval reward",
        tick_step=tick_step,
    )
    _plot_subfig(
        ax_ent,
        x_loss,
        entropy_loss,
        line_style="-",
        color="y",
        title="entropy loss",
        tick_step=tick_step,
        log_if_needed=True,
    )
    _plot_subfig(
        ax_critic,
        x_loss,
        critic_loss,
        line_style="-",
        color="g",
        title="critic loss",
        tick_step=tick_step,
        log_if_needed=True,
    )
    _plot_subfig(
        ax_actor,
        x_loss,
        actor_loss,
        line_style="-",
        color="b",
        title="actor loss",
        tick_step=tick_step,
    )
    if disc_loss is not None:
        _plot_subfig(
            ax_disc,
            x_loss,
            disc_loss,
            line_style="-",
            color="orange",
            title="discriminator loss",
            tick_step=tick_step,
            log_if_needed=True,
        )
        _plot_subfig(
            ax_disc_acc,
            x_disc,
            disc_acc,
            line_style="-",
            color="m",
            title="discriminator accuracy",
            tick_step=tick_step,
            label="actual",
        )
        _plot_subfig(
            ax_disc_prec,
            x_disc,
            disc_prec,
            line_style="-",
            color="m",
            title="discriminator precision",
            tick_step=tick_step,
            label="actual",
        )
        _plot_subfig(
            ax_disc_recall,
            x_disc,
            disc_recall,
            line_style="-",
            color="m",
            title="discriminator recall",
            tick_step=tick_step,
            label="actual",
        )

    if sma > 1:
        _plot_subfig(
            ax_train_sma,
            x_loss,
            train_reward,
            line_style="-",
            color="r",
            title="train reward",
            tick_step=tick_step,
            sma=sma,
        )
        if reward_sma > 1:
            _plot_subfig(
                ax_eval_sma,
                x_reward,
                eval_reward,
                line_style="-",
                color="r",
                title="eval reward",
                tick_step=tick_step,
                sma=reward_sma,
                sma_tick_extend_factor=eval_step,
            )
        _plot_subfig(
            ax_ent_sma,
            x_loss,
            entropy_loss,
            line_style="-",
            color="y",
            title="entropy loss",
            tick_step=tick_step,
            log_if_needed=True,
            sma=sma,
        )
        _plot_subfig(
            ax_critic_sma,
            x_loss,
            critic_loss,
            line_style="-",
            color="g",
            title="critic loss",
            tick_step=tick_step,
            log_if_needed=True,
            sma=sma,
        )
        _plot_subfig(
            ax_actor_sma,
            x_loss,
            actor_loss,
            line_style="-",
            color="b",
            title="actor loss",
            tick_step=tick_step,
            sma=sma,
        )
        if disc_loss is not None:
            _plot_subfig(
                ax_disc_sma,
                x_loss,
                disc_loss,
                line_style="-",
                color="orange",
                title="discriminator loss",
                tick_step=tick_step,
                log_if_needed=True,
                sma=sma,
            )
            _plot_subfig(
                ax_disc_acc,
                x_disc,
                disc_acc,
                line_style="-",
                color="black",
                tick_step=tick_step,
                label=f"SMA{sma}",
                sma=sma,
            )
            _plot_subfig(
                ax_disc_prec,
                x_disc,
                disc_prec,
                line_style="-",
                color="black",
                tick_step=tick_step,
                label=f"SMA{sma}",
                sma=sma,
            )
            _plot_subfig(
                ax_disc_recall,
                x_disc,
                disc_recall,
                line_style="-",
                color="black",
                tick_step=tick_step,
                label=f"SMA{sma}",
                sma=sma,
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
    input_y: List[float],
    line_style: str = "-",
    color: str = "r",
    title: Optional[str] = None,
    tick_step: int = 20,
    log_if_needed: bool = False,
    label: Optional[str] = None,
    sma: Optional[int] = None,
    sma_tick_extend_factor: int = 1,
):
    if sma is not None:
        assert isinstance(sma, int) and sma > 0
        assert isinstance(
            sma_tick_extend_factor, int
        ), "Wait until the first evaluation is done, and `plot.py` will be available."
        y = simple_moving_average(input_y, window=sma)
        if title is not None:
            title = f"SMA{sma * sma_tick_extend_factor} " + title
    else:
        y = input_y

    log_scale = False
    if log_if_needed and np.abs((max(y) + 1e-12) / (min(y) + 1e-12)) > 100:
        log_scale = True
        y = np.array(np.log10(np.array(y) + 1e-12)).tolist()

    # list[None:] == list
    ax.plot(
        x[(sma - 1) if sma is not None else 0 :],
        y[(sma - 1) if sma is not None else 0 :],
        line_style,
        color=color,
        label=label,
    )
    x_ticks = generate_absolute_ticks(1, max(x) if len(x) > 0 else 1, step=tick_step)

    if label is not None:
        ax.legend(loc="upper left")
    if title is not None:
        ax.set_title(("log10 " if log_scale else "") + title)
    if sma is not None:
        ax.axvline(sma * sma_tick_extend_factor, linestyle="--", color="black")

        x_ticks = set(x_ticks)
        x_ticks.add(sma * sma_tick_extend_factor)
        x_ticks = sorted(x_ticks)

    ax.set_xticks(x_ticks)


def simple_moving_average(vals: List[float], window: int = 10) -> List[float]:
    if window == 1 or len(vals) < window:
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

    if args.log.endswith("shrink.log"):
        args.plot = True
        args.cov = False
    elif args.log.endswith("cov.log"):
        args.plot = False
        args.cov = True

    log_filename = (
        get_latest_log(args.log) if not args.log.endswith(".log") else args.log
    )

    if args.eta > 0:
        get_eta(log_filename, args.eta)

    if args.plot:
        plot_filename_shrink_to = (
            osp.join(args.save, "shrink.log") if args.shrink and args.plot else None
        )
        train_reward, eval_reward, loss, eval_step, is_ac, disc_output = ocp_get_y_axis(
            log_filename, plot_filename_shrink_to
        )

        plot_subfig(
            is_ac,
            train_reward,
            eval_reward,
            loss,
            eval_step,
            to_filename(args.save, ".png", "reward_loss_fig"),
            tick_step=args.step,
            sma=args.sma,
            reward_sma=args.reward_sma,
            disc_output=disc_output,
        )
    if args.cov:
        cov_filename_shrink_to = (
            osp.join(args.save, "cov.log") if args.shrink and args.cov else None
        )
        train_covs, eval_covs, eval_step = ocp_get_cov(
            log_filename, cov_filename_shrink_to
        )

        plot_cov(
            train_covs,
            eval_covs,
            eval_step,
            to_filename(args.save, ".png", "cov_fig"),
            tick_step=args.step,
            sma=args.sma,
            eval_sma=args.reward_sma,
        )


if __name__ == "__main__":
    main()
