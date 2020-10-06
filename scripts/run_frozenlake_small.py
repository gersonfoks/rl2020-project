import numpy as np

from evaluation.mc import *
from frozen_lake_policies import FrozenLakeSmallPolicy
from utils.misc import *
from policies import *
import gym
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
policy = FrozenLakeSmallPolicy(actions, Q, 0.00001)

### First we run the env with random agent

observation = env.reset()
for _ in range(1000):
  env.render()
  action = policy.sample_action(observation) # your agent here (this takes random actions)
  print(action)
  observation, reward, done, info = env.step(action)
  if done:
    break
env.close()