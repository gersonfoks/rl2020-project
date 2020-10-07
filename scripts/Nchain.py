import numpy as np

from evaluation.mc import *
from utils.misc import *
from policies import *
import gym
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

###
n = 6


env = gym.make('NChain-v0')(n)


# Let's sample some episodes
actions = [0,1]
nchain_policy = RandomPolicy(actions)
policy = EpsilonGreedyPolicyNChain(actions)
for episode in range(3):
    trajectory_data = sample_episode(env, policy)
    print("Episode {}:\nStates {}\nActions {}\nRewards {}\nDones {}\n".format(episode,*trajectory_data))

np.random.seed(42)
V_10k = mc_ordinary_importance_sampling(env, nchain_policy, policy, 10000, sample_episode)
V_500k = mc_ordinary_importance_sampling(env, nchain_policy, policy, 500000, sample_episode)


Vs = [V_10k, V_500k]
x = np.arange(n)
fig = plt.figure(figsize=(20, 10))

for i in Vs:
    number_episodes = 1e4 if i == V_10k else 1e5
    plt.scatter(x,[i[j] for j in x], label= str(number_episodes))
plt.xlabel('States')
plt.ylabel('V')
plt.show()