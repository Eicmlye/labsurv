import copy
import numpy as np

from labsurv.builders import ALGORITHMS, ENVIRONMENTS


DANGER = 0
FREE = 1
START = 2
DEST = 3


INDENT = " " * 4


@ALGORITHMS.register_module()
class PolicyIterationAlgo:
  def __init__(self, env_cfg, gamma=0.9, threshold=0.001, converge_threshold=0.0001):
    self.env = ENVIRONMENTS.build(env_cfg)
    self.state_value = np.zeros(self.env.shape)
    
    self.strategy = dict()
    for index in range(self.env.shape[0]):
      for jndex in range(self.env.shape[1]):
        state = (index, jndex)
        self.strategy.update(
          {state: get_init_probs(self.env.world, get_neighbours(state))}
        )

    self.gamma = gamma
    self.threshold = threshold
    self.converge_threshold = converge_threshold
    self.epoch = 0

  def forward(self):
    old_strategy = dict()
    while not check_same_strategy(
      old_strategy, self.strategy, self.converge_threshold
    ):
      self.epoch += 1
      old_strategy = copy.deepcopy(self.strategy)
      self.policy_evaluation()
      self.policy_improvement()
    
    self.print_strategy()

  def policy_evaluation(self):
    max_diff = self.threshold  # any number no less than threshold

    count_iter = 0
    while max_diff >= self.threshold:
      max_diff = 0
      new_state_value = np.zeros(self.env.shape)
      for index in range(self.env.shape[0]):
        for jndex in range(self.env.shape[1]):
          cur_state = (index, jndex)
          for action in range(len(self.env.actions)):
            prob, next_state, reward, done = self.env.transition[cur_state][action]
            if done is not None:
              new_state_value[cur_state] += self.strategy[cur_state][action] * (
                reward + self.gamma * prob * self.state_value[next_state]
              )
          
          max_diff = max(
            max_diff, abs(new_state_value[cur_state] - self.state_value[cur_state])
          )
      
      self.state_value = new_state_value

      count_iter += 1
      print(
        f"[Policy Evaluation]  Epoch [{self.epoch}:{count_iter}] max_diff: {max_diff:.4f}"
      )

  def policy_improvement(self):
    for index in range(self.env.shape[0]):
      for jndex in range(self.env.shape[1]):
        cur_state = (index, jndex)
        state_action_value = np.array([-1000000] * len(self.env.actions))
        for action in range(len(self.env.actions)):
          prob, next_state, reward, done = self.env.transition[cur_state][action]
          if done is not None:
            state_action_value[action] = reward + self.gamma * prob * self.state_value[
              next_state
            ]
        max_value = np.max(state_action_value)
        unique, counts = np.unique(state_action_value, return_counts=True)
        max_count = dict(zip(unique, counts))[max_value]
        self.strategy[cur_state] = [
          1 / max_count 
          if state_action_value[action] == max_value 
          else 0 
          for action in range(len(self.env.actions))
        ]

    print(f"[Policy Improvement] Epoch [{self.epoch}].")

  def print_strategy(self):
    output = [[""] * self.env.shape[1] for _ in range(self.env.shape[0])]
    print("======== value ========")
    for index in range(self.env.shape[0]):
      for jndex in range(self.env.shape[1]):
        state = (index, jndex)
        if self.env.world[state] == DANGER:
          output[index][jndex] = "XXXX"
        elif self.env.world[state] == DEST:
          output[index][jndex] = "YYYY"
        else:
          action_probs = self.strategy[state]
          max_value = max(action_probs)
          output[index][jndex] += "^" if action_probs[0] == max_value else "o"
          output[index][jndex] += "<" if action_probs[1] == max_value else "o"
          output[index][jndex] += "v" if action_probs[2] == max_value else "o"
          output[index][jndex] += ">" if action_probs[3] == max_value else "o"
        
        print(f"{self.state_value[state]:>8.4f}", end="\n" if jndex == self.env.shape[1] - 1 else INDENT)

    print("======== strategy ========")
    for row in output:
      for grid in row:
        print(f"{grid}", end=INDENT)
      
      print("")


def get_neighbours(pos):
  index, jndex = pos

  neighbours = [
    (index - 1, jndex),  # north
    (index, jndex - 1),  # west
    (index + 1, jndex),  # south
    (index, jndex + 1),  # east
  ]

  return neighbours


def get_init_probs(world, neighbours):
  if world[neighbours[0][0] + 1, neighbours[0][1]] in [DANGER, DEST]:
    return [0] * 4
  valid_neighbours = []
  for neighbour in neighbours:
    if neighbour[0] in range(world.shape[0]) and neighbour[1] in range(world.shape[1]):
      valid_neighbours.append(1)
    else:
      valid_neighbours.append(0)

  return np.array(valid_neighbours) / sum(valid_neighbours)


def check_same_strategy(old_strat, new_strat, threshold):
  if old_strat.keys() != new_strat.keys():
    return False

  for state in old_strat.keys():
    if len(old_strat[state]) != len(new_strat[state]):
      return False
    for index in range(len(old_strat[state])):
      if abs(old_strat[state][index] - new_strat[state][index]) >= threshold:
        return False
    
  return True


if __name__ == "__main__":
  from mmengine import Config
  from labsurv.builders import ENVIRONMENTS

  cfg_path = input("Enter config path: ")
  if cfg_path == "":
    cfg_path = "configs/cliff_walk/cliff_walk.py"
  cfg = Config.fromfile(cfg_path)
  cliff_walk = ENVIRONMENTS.build(cfg.env)
  for pos, actions in cliff_walk.transition.items():
    print(f"{pos}: {np.array(cfg.world)[pos]}")
    for action, info in actions.items():
      print(f"    {action.name}:{info}")