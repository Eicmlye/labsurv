import os
import os.path as osp
from collections import deque
from datetime import datetime
from typing import Dict

from labsurv.builders import HOOKS
from labsurv.runners.hooks.utils import (
    get_cur_time_str,
    get_episode_progress_str,
    get_latest_avg_reward_str,
    get_log_str,
    get_time_eta_strs,
    merge_log_str,
)
from labsurv.utils import INDENT, get_time_stamp


@HOOKS.register_module()
class LoggerHook:
    def __init__(self, log_interval: int, save_dir: str, save_filename: str = None):
        self.build_time = datetime.now()
        self.time = datetime.now()
        self.interval = log_interval
        self.log_file = osp.join(
            save_dir,
            (get_time_stamp() if save_filename is None else save_filename) + ".log",
        )
        self.return_list: deque = deque(maxlen=2 * log_interval)
        self.start_episode_index = 0
        self.cur_episode_index = -1
        self.save_dir = save_dir

        os.makedirs(save_dir, exist_ok=True)

    def set_start_episode_index(self, episode_index: int):
        self.start_episode_index = episode_index

    def set_cur_episode_index(self, episode_index: int):
        self.cur_episode_index = episode_index

    def show_log(self, log_str: str, with_time: bool = False, end: str = None):
        if end is None:
            end = "\n"
        log_str = (get_cur_time_str()[0] + INDENT + log_str) if with_time else log_str
        print(log_str, end=end)
        with open(self.log_file, "a+") as f:
            f.write(log_str + end)

    def update(self, return_val: float | Dict[str, float]):
        self.return_list.append(return_val)
        self.cur_episode_index += 1

    def __call__(self, episodes: int, **kwargs):
        self.time = datetime.now()

        if (self.cur_episode_index + 1) % self.interval == 0:
            log_list = []
            log_list += get_cur_time_str()
            log_list += get_episode_progress_str(self.cur_episode_index, episodes)
            log_list += get_time_eta_strs(
                self.build_time,
                self.time,
                self.start_episode_index,
                self.cur_episode_index,
                episodes,
            )
            for key in kwargs.keys():
                log_list += get_log_str(key, kwargs[key])

            log_list += get_latest_avg_reward_str(
                self.interval, self.return_list, show_returns=True
            )
            self.show_log(merge_log_str(log_list))
