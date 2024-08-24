import argparse
import os
import os.path as osp

from labsurv.runners import StepBasedRunner
from mmcv import Config


def parse_args():
    parser = argparse.ArgumentParser(description="RL trainer.")

    parser.add_argument("--config", type=str, help="Path of the config file.")
    parser.add_argument("--debug", action="store_true", help="Debug mode.")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    cfg.use_replay_buffer = "replay_buffer" in cfg.keys()

    os.makedirs(cfg.work_dir, exist_ok=True)
    save_cfg_name = osp.join(cfg.work_dir, cfg.exp_name + ".py")
    cfg.dump(save_cfg_name)

    runner = StepBasedRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
