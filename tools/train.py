import torch
import argparse
import os
import os.path as osp
from copy import deepcopy

from labsurv.builders import RUNNERS
from labsurv.runners import EpisodeBasedRunner, StepBasedRunner
from mmcv import Config


def parse_args():
    parser = argparse.ArgumentParser(description="RL trainer.")

    parser.add_argument("--config", type=str, help="Path of the config file.")
    parser.add_argument("--debug", action="store_true", help="Debug mode.")
    parser.add_argument(  # "--episode-based"
        "--episode-based",
        action="store_true",
        help="Whether to use an episode based runner.",
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    cfg.use_replay_buffer = "replay_buffer" in cfg.keys()

    os.makedirs(cfg.work_dir, exist_ok=True)
    save_cfg_name = osp.join(cfg.work_dir, cfg.exp_name + ".py")

    # NOTE(eric): mmcv.Config dump() method cannot treat DEVICE="cuda:0" correctly as a
    # string, instead, the item will be written as DEVICE=cuda:0, which raises
    # yapf_third_party._ylib2to3.pgen2.parse.ParseError: bad input: type=11, value=':', context=('', (1, 11))
    # when parsing the config file text. Therefore, here we hack the item and after dumping
    # the item is recovered.

    cfg.dump(save_cfg_name)

    if "runner" in cfg.keys():
        other_cfg = deepcopy(cfg)
        other_cfg.pop("runner")
        runner_cfg = dict(
            type=cfg.runner["type"],
            cfg=other_cfg,
        )

        runner = RUNNERS.build(runner_cfg)
    else:
        episode_based = (
            cfg.episode_based if hasattr(cfg, "episode_based") else args.episode_based
        )
        runner = EpisodeBasedRunner(cfg) if episode_based else StepBasedRunner(cfg)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    runner.run()


if __name__ == "__main__":
    main()
