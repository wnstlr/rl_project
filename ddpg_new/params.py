# ==========================
#   Training Parameters
# ==========================
# Max training steps
MAX_EPISODES = 50000
# Max episode length
MAX_EP_STEPS = 1000
MOMENTUM = 0.95
MOMENTUM_2 = 0.999
MAX_NORM = 10
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
# Discount factor
GAMMA = 0.99
# Soft target update param
TAU = 0.001
EXPLORE_ITER = 100
BETA = 0.5
STATE_INPUT_COUNT = 1


# ===========================
#   Utility Parameters
# ===========================
# Render gym env during training
RENDER_ENV = True
# Use Gym Monitor
GYM_MONITOR_EN = True
# Gym environment
ENV_NAME = 'Pendulum-v0'
# Directory for storing gym results
MONITOR_DIR = './results/gym_ddpg'
# Directory for storing tensorboard summary results
SUMMARY_DIR = './results/tf_ddpg'
# RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 500000
MEMORY_THRESHOLD = 1000
MINIBATCH_SIZE = 32
ACTION_SIZE = 4
PARAM_SIZE = 6
ACTION_INPUT_DATA_SIZE = MINIBATCH_SIZE * ACTION_SIZE
ACTION_PARAMS_INPUT_DATA_SIZE = MINIBATCH_SIZE * PARAM_SIZE
TARGET_INPUT_DATA_SIZE = MINIBATCH_SIZE * ACTION_SIZE
FILTER_INPUT_DATA_SIZE = MINIBATCH_SIZE * ACTION_SIZE

CONFIG_DIR = "bin/teams/base/config/formations-dt"
SERVER_ADDR = "localhost"
TEAM_NAME = "base_left"
PLAY_GOALIE = False
RECORD_DIR = ""

# ===============
# Params for main
# ===============
MAX_ITER = 10000000
UPDATE_RATIO = 0.1

EPSILON = 0.1
OFFENSE_AGENTS = 1
OFFENSE_NPCS = 0
DEFENSE_AGENTS = 0
DEFENSE_NPCS = 0
OFFENSE_DUMMIES = 0
DEFENSE_DUMMIES = 0
DEFENSE_CHASERS = 0

# Specify which GymSoccer env to use:
# soccer, socceragainstkeeper, socceremptygoal
ENVTYPE = 'soccer'

# If true, train on a simpler network structure
SIMPLE = True

# If true, continue from the last checkpoint
CONTINUE = False

# If true, test the model.
TEST = False
