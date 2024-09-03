import time
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Dict, List

from labsurv.utils import INDENT, WARN
from torch import Tensor


def get_cur_time_str() -> List[str]:
    cur_time = time.localtime()
    return [
        "["
        f"{cur_time.tm_year}-{cur_time.tm_mon:02d}-{cur_time.tm_mday:02d}"
        f" {cur_time.tm_hour:02d}:{cur_time.tm_min:02d}:{cur_time.tm_sec:02d}"
        "]"
    ]


def get_delta_str(delta: timedelta) -> str:
    days = delta.days
    hours = delta.seconds // 3600
    minutes = delta.seconds // 60 - hours * 60
    seconds = delta.seconds - hours * 3600 - minutes * 60

    return_str = ""
    if days != 0:
        return_str += f"{days} "
        return_str += "days " if days > 1 else "day "
    return_str += f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    return return_str


def get_episode_progress_str(episode_index: int, episodes: int) -> List[str]:
    assert episode_index < episodes, (
        f"Episode index {episode_index} should be "
        f"less than total episode number {episodes}."
    )

    episode = episode_index + 1
    return [f"Episode [{episode}/{episodes}]"]


def get_time_eta_strs(
    start_time: datetime, cur_time: datetime, episode_index: int, episodes: int
) -> List[str]:
    assert episode_index < episodes, (
        f"Episode index {episode_index} should be "
        f"less than total episode number {episodes}."
    )

    delta = cur_time - start_time
    episode = episode_index + 1
    avg_interval = delta / episode

    return [
        f"elapsed: {get_delta_str(delta)}",
        f"eta: {get_delta_str(avg_interval * (episodes - episode))}",
    ]


def get_latest_avg_reward_str(
    interval: int, return_list: deque, show_returns: bool = False
) -> List[str]:
    if not isinstance(return_list[0], Dict):
        raise ValueError("`return_list` type is rewritten, the code should be updated.")
    return_dict = dict()
    for item in return_list:
        for key, val in item.items():
            if key not in return_dict.keys():
                return_dict[key] = [val]
            else:
                return_dict[key].append(val)

    return_str = []

    if interval > len(return_dict["reward"]):
        print(
            WARN(
                f"`return_list` has not got enough values to be averaged. "
                f"Expected {interval}, got {len(return_dict["reward"])}."
            )
        )
    else:
        return_str.append(
            f"avg reward: {sum(return_dict["reward"][-interval:]) / interval:.4f}"
        )

        if show_returns:
            return_str.append(f"\nlast {interval} returns: ")
            for key, val in return_dict.items():
                return_str.append(f"\n\t{key:<12}: [")
                if "cov" in key:
                    return_str[-1] += f"{return_dict[key][-interval]:.2%}"
                    for item in return_dict[key][-interval + 1 :]:
                        return_str[-1] += f", {item:.2%}"
                else:
                    return_str[-1] += f"{return_dict[key][-interval]:.4f}"
                    for item in return_dict[key][-interval + 1 :]:
                        return_str[-1] += f", {item:.4f}"
                return_str[-1] += "]"

    return return_str


def get_log_str(key: str, val: Any) -> List[str]:
    if isinstance(val, float):
        val = f"{val:.4e}"
    elif isinstance(val, Tensor) and val.ndim == 0:
        val = f"{val:.4f}"
    elif isinstance(val, Tensor) and len(val.shape) == 1 and val.shape[0] == 1:
        val = f"{val.item():.4f}"

    return [f"{key}: {val}"]


def merge_log_str(log_list: List[str], separator: str = INDENT):
    return separator.join(log_list)
