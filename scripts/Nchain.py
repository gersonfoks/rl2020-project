import numpy as np

from evaluation.mc import *
from utils.misc import *
from policies import *
import gym
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gym.envs.registration import register

n = 7

Q = np.zeros((n, 2))

for state in range(n):
    Q[state,  0] = 1

register( id='NChainN-v0', entry_point='gym.envs.toy_text:NChainEnv', kwargs={'n': n} )
env = gym.make('NChainN-v0')
env.n = n
print(env.action_space)
print(Q)

# Let's sample some episodes
actions = [0,1]
target_policy = EpsilonGreedyPolicy(actions, Q, 0.1)
behavior_policy = RandomPolicy(actions)

for episode in range(3):
    trajectory_data = sample_episode(env, behavior_policy)
    print("Episode {}:\nStates {}\nActions {}\nRewards {}\nDones {}\n".format(episode,*trajectory_data))

np.random.seed(42)
V_10k = mc_weighted_importance_sampling(env,behavior_policy , target_policy, 10000, sample_episode)
# V_500k = mc_weighted_importance_sampling(env, behavior_policy, target_policy, 500000, sample_episode)

# Vs = [V_10k, V_500k]
# x = np.arange(n)
# fig = plt.figure(figsize=(20, 10))

# for i in Vs:
#     number_episodes = 1e4 if i == V_10k else 1e5
#     plt.scatter(x,[i[j] for j in x], label= str(number_episodes))
# plt.xlabel('States')
# plt.ylabel('V')
# plt.show()





print(V_10k)