import argparse
import os
import os.path as osp
import random

import matplotlib.pyplot as plt
import numpy as np
from labsurv.builders import AGENTS, ENVIRONMENTS, HOOKS, REPLAY_BUFFERS
from mmengine import Config


def parse_args():
    parser = argparse.ArgumentParser(description="DQN trainer.")

    parser.add_argument("--config", type=str, help="Path of the config file.")
    parser.add_argument("--debug", action="store_true", help="debug mode")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    np.random.seed(0)
    random.seed(0)

    env = ENVIRONMENTS.build(cfg.env)
    agent = AGENTS.build(cfg.agent)
    logger = HOOKS.build(cfg.logger_cfg)

    cfg.use_replay_buffer = "replay_buffer" in cfg.keys()
    if cfg.use_replay_buffer:
        replay_buffer = REPLAY_BUFFERS.build(cfg.replay_buffer)

    os.makedirs(cfg.work_dir, exist_ok=True)
    save_cfg_name = osp.join(cfg.work_dir, cfg.exp_name + ".py")
    cfg.dump(save_cfg_name)

    episodes = cfg.episodes
    return_list = []

    cur_state = None
    batch_size = cfg.batch_size
    for episode in range(episodes):
        cur_state = env.reset()[0]

        episode_return = 0
        terminated = False
        truncated = False

        while not terminated and not truncated:
            cur_action = agent.take_action(cur_state)
            next_state, reward, terminated, truncated, _ = env.step(cur_action)
            if cfg.use_replay_buffer:
                replay_buffer.add(
                    cur_state,
                    cur_action,
                    reward,
                    next_state,
                    terminated,
                    truncated,
                )
            cur_state = next_state
            episode_return += reward

            if cfg.use_replay_buffer and replay_buffer.is_active():
                transitions = replay_buffer.sample(batch_size)
                agent.update(**transitions)

        logger.update(episode_return)
        logger(episodes)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(f"{agent.dqn_type} on 'Cart Pole'")
    plt.show()


if __name__ == "__main__":
    main()
