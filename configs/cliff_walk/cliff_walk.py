DANGER = 0
FREE = 1
START = 2
DEST = 3

# world = [
#   [FREE,    FREE,     FREE,     FREE,     FREE,     FREE],
#   [FREE,    FREE,     FREE,     FREE,     FREE,     FREE],
#   [FREE,    FREE,     FREE,     FREE,     FREE,     FREE],
#   [START,   DANGER,   DANGER,   DANGER,   DANGER,   DEST],
# ]

# world = [
#   [FREE,    FREE,     FREE,     FREE,     FREE,     FREE],
#   [FREE,    DANGER,   FREE,     DANGER,   DANGER,   DANGER],
#   [FREE,    DANGER,   FREE,     FREE,     FREE,     FREE],
#   [START,   DANGER,   FREE,     DANGER,   DANGER,   DEST],
# ]

# world = [
#   [FREE,    FREE,     FREE,     FREE,     FREE,     START],
#   [FREE,    DANGER,   FREE,     DANGER,   DANGER,   DANGER],
#   [FREE,    DANGER,   FREE,     FREE,     FREE,     FREE],
#   [FREE,    DANGER,   FREE,     DANGER,   DANGER,   DEST],
# ]

world = [
  [FREE,    FREE,     FREE,     FREE,     FREE,     FREE],
  [FREE,    DANGER,   FREE,     DANGER,   DANGER,   DANGER],
  [FREE,    DANGER,   FREE,     FREE,     FREE,     FREE],
  [START,   DANGER,   FREE,     DANGER,   DANGER,   DEST],
  [FREE,    DANGER,   FREE,     FREE,     FREE,     DANGER],
  [FREE,    FREE,     FREE,     DANGER,   FREE,     FREE],
]

# world = [
#   [FREE,    FREE,     FREE,     FREE,     FREE,     FREE],
#   [FREE,    DANGER,   FREE,     DANGER,   FREE,     FREE],
#   [FREE,    DANGER,   FREE,     DANGER,   FREE,     START],
#   [FREE,    DANGER,   FREE,     DANGER,   DANGER,   DANGER],
#   [FREE,    DANGER,   FREE,     FREE,     FREE,     FREE],
#   [FREE,    DANGER,   FREE,     FREE,     FREE,     FREE],
#   [FREE,    DANGER,   FREE,     FREE,     FREE,     FREE],
#   [FREE,    DANGER,   FREE,     DANGER,   DANGER,   DEST],
# ]

# world = [
#   [FREE,     FREE,     FREE,     FREE,     FREE,     FREE],
#   [FREE,     DANGER,   FREE,     FREE,     DANGER,   FREE],
#   [DANGER,   DANGER,   FREE,     FREE,     FREE,     FREE],
#   [START,    FREE,     DANGER,   DANGER,   FREE,     FREE],
#   [FREE,     DANGER,   FREE,     DANGER,   FREE,     DEST],
#   [FREE,     DANGER,   DANGER,   FREE,     FREE,     FREE],
#   [FREE,     FREE,     FREE,     FREE,     FREE,     FREE],
#   [FREE,     FREE,     FREE,     DANGER,   FREE,     FREE],
# ]

# world = [
#   [FREE,     FREE,     FREE,     FREE,     FREE,     FREE],
#   [FREE,     FREE,     FREE,     DANGER,   FREE,     FREE],
#   [FREE,     DANGER,   FREE,     DANGER,   FREE,     FREE],
#   [START,    DANGER,   FREE,     DANGER,   FREE,     FREE],
#   [FREE,     DANGER,   FREE,     DANGER,   DANGER,   DANGER],
#   [FREE,     DANGER,   FREE,     FREE,     FREE,     FREE],
#   [FREE,     DANGER,   FREE,     DANGER,   DANGER,   FREE],
#   [FREE,     DANGER,   FREE,     FREE,     DEST,     FREE],
# ]

# world = [
#   [FREE,     FREE,     FREE,     FREE,     FREE,     FREE],
#   [FREE,     FREE,     FREE,     DANGER,   FREE,     FREE],
#   [FREE,     DANGER,   FREE,     DANGER,   FREE,     FREE],
#   [FREE,     DANGER,   FREE,     DANGER,   FREE,     START],
#   [FREE,     DANGER,   FREE,     DANGER,   DANGER,   DANGER],
#   [FREE,     DANGER,   FREE,     FREE,     FREE,     FREE],
#   [FREE,     DANGER,   FREE,     DANGER,   DANGER,   FREE],
#   [FREE,     DANGER,   FREE,     FREE,     DEST,     FREE],
# ]

env = dict(
  type="CliffWalkModelFreeEnv",
  world=world,
  reward={
    DANGER: -50,
    FREE: -1,
    START: -1,
    DEST: 50,
  },
)

algo = dict(
  type="QLearningAlgo",
  world=world,
  action_num=4,
  lr=0.1,
  gamma=0.9,
  greedy_epsilon=0.2,
  step=1,
)

episodes = 10000
log_interval = 100
