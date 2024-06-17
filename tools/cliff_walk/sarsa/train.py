import argparse

import matplotlib.pyplot as plt
from labsurv.builders import AGENTS, ENVIRONMENTS
from mmengine.config import Config


def parse_args():
    parser = argparse.ArgumentParser(description="RL trainer.")

    parser.add_argument("--config", type=str, help="Path of the config file.")
    parser.add_argument("--debug", action="store_true", help="debug mode")

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
        if args.debug:
            agent.print_strategy()
            agent.print_qtable()
            input()
        cur_state = env.reset()

        episode_return = 0
        terminated = False

        cur_action = agent.take_action(cur_state)
        while not terminated:
            next_state, reward, terminated = env.step(cur_action)
            episode_return += reward
            next_action = agent.take_action(next_state)
            if args.debug:
                agent.print_strategy()
                agent.print_qtable()
                print(f"cur_action = {cur_action}, next_action = {next_action}")
                # import pdb; pdb.set_trace()  # EM
            agent.update(
                cur_state,
                cur_action,
                reward,
                next_state,
                next_action,
                terminated,
            )

            cur_state = next_state
            cur_action = next_action
        return_list.append(episode_return)
        if (episode + 1) % cfg.log_interval == 0:
            print(
                f"\rEpisode [{episode + 1}/{episodes}] "
                f"reward: {sum(return_list[-cfg.log_interval:]) / cfg.log_interval}\033[K"
            )
            agent.print_strategy()

    episodes_list = list(range(len(return_list)))
    agent.print_strategy()
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('SARSA on {}'.format('Cliff Walking'))
    plt.show()


if __name__ == "__main__":
    main()
