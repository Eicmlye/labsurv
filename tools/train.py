import argparse
import os
import os.path as osp

from labsurv.runners import EpisodeBasedRunner, StepBasedRunner
from mmcv import Config


def parse_args():
    parser = argparse.ArgumentParser(description="RL trainer.")

    parser.add_argument("--config", type=str, help="Path of the config file.")
    parser.add_argument("--debug", action="store_true", help="Debug mode.")
    parser.add_argument(  # "--episode-based"
        "--episode-based", action="store_true", help="Whether use episode based runner."
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    cfg.use_replay_buffer = "replay_buffer" in cfg.keys()

    os.makedirs(cfg.work_dir, exist_ok=True)
    save_cfg_name = osp.join(cfg.work_dir, cfg.exp_name + ".py")
    cfg.dump(save_cfg_name)

    episode_based = (
        cfg.episode_based if hasattr(cfg, "episode_based") else args.episode_based
    )

    runner = EpisodeBasedRunner(cfg) if episode_based else StepBasedRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
