import numpy as np

from evaluation.td import *
from utils.misc import *
from policies import *
import gym
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

###
player_values = [i for i in range(12, 22)]
dealer_values = [i for i in range(1,11)]


env = gym.make('Blackjack-v0')


# Let's sample some episodes

policy = SimpleBlackjackPolicy()
for episode in range(3):
    trajectory_data = sample_episode(env, policy)
    print("Episode {}:\nStates {}\nActions {}\nRewards {}\nDones {}\n".format(episode,*trajectory_data))


actions = [0,1]
blackjack_policy = RandomPolicy(actions)

# V_10k = td_n(env, SimpleBlackjackPolicy(), 0.01, 10000, sample_step, n=3)
# V_500k = td_n(env, SimpleBlackjackPolicy(), 0.01, 500000, sample_step, n=3)
np.random.seed(42)
V_10k = n_step_td_ordinary_importance_sampling(env, blackjack_policy, SimpleBlackjackPolicy(), 10000, sample_step)
V_500k = n_step_td_ordinary_importance_sampling(env, blackjack_policy, SimpleBlackjackPolicy(), 500000, sample_step)


usable_aces_values = [True, False]
Vs = [V_10k, V_500k]
X, Y = np.meshgrid(player_values, dealer_values)
fig = plt.figure(figsize=(20, 10))
c = 0
for i, usable_ace in enumerate(usable_aces_values):
    for j, V in enumerate(Vs):
        c += 1

        number_episodes = 1e4 if V == V_10k else 5e5

        predicted_values = get_predicted_values(V, player_values, dealer_values, usable_ace)
        ax = fig.add_subplot(2, 2, c, projection='3d')
        plt.xlabel('Dealer Showing')
        plt.ylabel('Player sum')
        plt.title("Usable ace: {}, after {:,} episodes".format(usable_ace, int(number_episodes)))
        ax.plot_wireframe(Y, X, predicted_values, linewidth=0.2, antialiased=False)
plt.show()