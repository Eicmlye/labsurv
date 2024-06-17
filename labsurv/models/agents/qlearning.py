import numpy as np
from labsurv.builders import AGENTS
from labsurv.utils import INDENT

DANGER = 0
FREE = 1
START = 2
DEST = 3


@AGENTS.register_module()
class QLearningAgent:
    def __init__(
        self, world, action_num, lr=0.1, gamma=0.9, greedy_epsilon=0.05, step=1
    ):
        self._world = world
        self._shape = (len(world), len(world[0]))
        self.qtable = np.zeros((self._shape[0] * self._shape[1], action_num))
        self.action_num = action_num
        self.lr = lr
        self.gamma = gamma
        self.epsilon = greedy_epsilon
        self.step = step
        self.memory = dict(
            state_index=[],
            action=[],
            reward=[],
        )

    def take_action(self, state):
        if np.random.random() < self.epsilon:  # explore
            return np.random.randint(self.action_num)
        else:  # exploit
            state_index = self.get_state_index(state)
            max_value = np.max(self.qtable[state_index])

            unique, counts = np.unique(self.qtable[state_index], return_counts=True)

            out_action_index_among_max = np.random.randint(
                dict(zip(unique, counts))[max_value]
            )
            for index in range(self.action_num):
                if self.qtable[state_index][index] == max_value:
                    if out_action_index_among_max == 0:
                        return index
                    out_action_index_among_max -= 1

    def update(self, cur_state, cur_action, reward, next_state, terminated):
        cur_state_index = self.get_state_index(cur_state)
        next_state_index = self.get_state_index(next_state)
        self.memory["state_index"].append(cur_state_index)
        self.memory["action"].append(cur_action)
        self.memory["reward"].append(reward)

        if len(self.memory["state_index"]) == self.step:
            total_reward = max(self.qtable[next_state_index])
            for step in reversed(range(self.step)):
                total_reward = self.gamma * total_reward + self.memory["reward"][step]

                # update the remaining short chains when terminated
                if terminated and step > 0:
                    state_index = self.memory["state_index"][step]
                    action = self.memory["action"][step]
                    self.qtable[state_index, action] += self.lr * (
                        total_reward - self.qtable[state_index, action]
                    )

            state = self.memory["state_index"].pop(0)
            action = self.memory["action"].pop(0)
            self.memory["reward"].pop(0)

            self.qtable[state, action] += self.lr * (
                total_reward - self.qtable[state, action]
            )

        if terminated:
            self.memory = dict(
                state_index=[],
                action=[],
                reward=[],
            )

    def get_state_index(self, state):
        return self._shape[1] * state[0] + state[1]

    def print_strategy(self):
        print("\n======== strategy ========")
        for index in range(self._shape[0]):
            for jndex in range(self._shape[1]):
                state_index = self.get_state_index((index, jndex))
                max_value = np.max(self.qtable[state_index])
                if self._world[index][jndex] == DANGER:
                    strategy_str = "XXXX"
                elif self._world[index][jndex] == DEST:
                    strategy_str = "YYYY"
                else:
                    strategy_str = ""
                    strategy_str += (
                        "^" if self.qtable[state_index][0] == max_value else "o"
                    )
                    strategy_str += (
                        "<" if self.qtable[state_index][1] == max_value else "o"
                    )
                    strategy_str += (
                        "v" if self.qtable[state_index][2] == max_value else "o"
                    )
                    strategy_str += (
                        ">" if self.qtable[state_index][3] == max_value else "o"
                    )

                print(strategy_str, end=INDENT)
            print("")

    def print_qtable(self):
        print("======== qtable ========")
        for index in range(self._shape[0]):
            for jndex in range(self._shape[1]):
                state_index = self.get_state_index((index, jndex))
                print("[", end="")
                for action in range(self.action_num):
                    print(f"{self.qtable[state_index][action]:>4.2f}", end=" " * 2)
                print("\b\b]", end="")

            print("")
