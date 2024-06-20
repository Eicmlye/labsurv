import os
import os.path as osp
from datetime import datetime

from labsurv.builders import HOOKS
from labsurv.runners.hooks.utils import (
    get_cur_time_str,
    get_episode_progress_str,
    get_latest_avg_reward_str,
    get_time_eta_strs,
    merge_log_str,
)
from labsurv.utils import get_time_stamp


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
        self.return_list = []

        os.makedirs(save_dir, exist_ok=True)

    def show_log(self, log_str: str):
        print(log_str)
        with open(self.log_file, "a+") as f:
            f.write(log_str + "\n")

    def update(self, return_val):
        self.return_list.append(return_val)

    def __call__(self, episodes: int):
        episode = len(self.return_list)
        self.time = datetime.now()

        if (episode + 1) % self.interval == 0:
            log_list = []
            log_list += get_cur_time_str()
            log_list += get_episode_progress_str(episode, episodes)
            log_list += get_time_eta_strs(self.build_time, self.time, episode, episodes)
            log_list += get_latest_avg_reward_str(
                self.interval, self.return_list, show_returns=True
            )
            self.show_log(merge_log_str(log_list))
