from evaluation.td import n_step_td_off_policy
from frozen_lake_policies import FrozenLakeSmallPolicy
from utils.misc import *
from policies import *
import gym


## We will use a non slippery variant First.
from gym.envs.registration import register
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)

env = gym.make('FrozenLakeNotSlippery-v0')

env.is_slippery = False

print(env.action_space)


action_map = {
  0: "D",
  1: "R",
  2: "D",
  3: "L",
  4: "D",
  5: "U",
  6: "D",
  7: "U",
  8: "R",
  9: "D",
  10: "D",
  11: "U",
  12: "U",
  13: "R",
  14: "R",
  15: "D",

}
index = {"L": 0, "D": 1, "R": 2, "U": 3}

Q = np.zeros((16, 4))

for state, action in action_map.items():
  Q[state,  index[action]] =1

print(Q)

actions = [0,1,2,3]
target_policy = FrozenLakeSmallPolicy(actions, Q, 0.000001)
behavior_policy = RandomPolicy(actions)

### First we run the env with random agent

np.random.seed(42)
#V_10k = mc_weighted_importance_sampling(env,behavior_policy , target_policy, 10000, sample_episode)
V_500k = n_step_td_off_policy(env, behavior_policy, target_policy, 100000, sample_step, n=3)

print(V_500k)