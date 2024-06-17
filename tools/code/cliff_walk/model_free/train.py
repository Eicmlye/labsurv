import argparse

from labsurv.builders import AGENTS, ENVIRONMENTS
import matplotlib.pyplot as plt
from mmengine.config import Config


def parse_args():
    parser = argparse.ArgumentParser(description="RL trainer.")

    parser.add_argument("--config", type=str, help="Path of the config file.")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    env = ENVIRONMENTS.build(cfg.env)
    agent = AGENTS.build(cfg.agent)

    episodes = cfg.episodes
    return_list = []

    cur_state = None
    for episode in range(episodes):
        cur_state = env.reset()

        episode_return = 0
        terminated = False

        cur_action = agent.take_action(cur_state)
        while not terminated:
            next_state, reward, terminated = env.step(cur_action)
            episode_return += reward
            next_action = agent.take_action(next_state)
            agent.update(cur_state, cur_action, reward, next_state, next_action)

            cur_state = next_state
            cur_action = next_action
        return_list.append(episode_return)
        if (episode + 1) % cfg.log_interval == 0:
            print(
                f"\rEpisode [{episode + 1}/{episodes}] reward: {episode_return}\033[K"
            )

    episodes_list = list(range(len(return_list)))
    agent.print_strategy()
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('SARSA on {}'.format('Cliff Walking'))
    plt.show()


if __name__ == "__main__":
    main()
