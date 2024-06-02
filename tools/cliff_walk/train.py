import argparse

from labsurv.builders import ALGORITHMS
from mmengine.config import Config


def parse_args():
  parser = argparse.ArgumentParser(description="RL trainer.")

  parser.add_argument("--config", type=str, help="Path of the config file.")

  args = parser.parse_args()

  return args


def main():
  args = parse_args()

  cfg = Config.fromfile(args.config)

  agent = ALGORITHMS.build(cfg.algo)
  agent.forward()


if __name__ == "__main__":
  main()