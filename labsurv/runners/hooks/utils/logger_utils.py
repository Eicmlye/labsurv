import time
from datetime import datetime, timedelta
from typing import List

from labsurv.utils import INDENT


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
    interval: int, return_list: list, show_returns: bool = False
) -> List[str]:
    assert interval <= len(return_list), (
        f"return_list has not got enough values to be averaged."
        f"Expected {interval}, got {len(return_list)}."
    )

    return_str = [f"avg reward: {sum(return_list[-interval:]) / interval:.4f}"]
    if show_returns:
        return_str.append(f"last {interval} returns: {return_list[-interval:]}")

    return return_str


def merge_log_str(log_list: List[str], separator: str = INDENT):
    return separator.join(log_list)
