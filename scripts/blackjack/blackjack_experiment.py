"""
This file contains the blackjack experiments.
"""
import gym
import numpy as np
from evaluation.mc import *
from evaluation.td import n_step_td_off_policy
from utils.experiments import *
from utils.misc import sample_episode, sample_step
from policies import SimpleBlackjackPolicy, RandomPolicy
import matplotlib.pyplot as plt

# Default variables

alphas = [0.01]
td_n = 4

actions = [0, 1]
env = gym.make('Blackjack-v0')

# Global settings

target_policy = SimpleBlackjackPolicy()
behavior_policy = RandomPolicy(actions)

n_experiments = 10

save_every = 1e3  ### How often we should save the results

# Conf for mc
n_mc_run = int(5e5)
save_every_mc = n_mc_run

# Conf for the mc off policy

n_mc_off_policy = int(5e5)

# First we need to run mc.
v_mc, hist = mc_prediction(env, SimpleBlackjackPolicy(), n_mc_run, sample_episode, save_every=n_mc_run, name="mc_blackjack")

histories_ord = run_experiments(mc_ordinary_importance_sampling, env, behavior_policy, target_policy, n_mc_off_policy,
                                sample_episode, n_experiments, save_every, name="mc_ord_blackjack")

histories_weighted = run_experiments(mc_weighted_importance_sampling, env, behavior_policy, target_policy,
                                     n_mc_off_policy, sample_episode, n_experiments, save_every,
                                     name="mc_weighted_blackjack")


alpha_histories = [

]
for alpha in alphas:
    alpha_histories.append(run_experiments(n_step_td_off_policy, env, behavior_policy, target_policy, n_mc_off_policy, sample_step,
                               n_experiments, save_every, name="td_blackjack_{}".format(alpha), alpha=alpha, n=td_n))

# Next we plot the results.

list_of_histories = [
    histories_ord,
    histories_weighted,

] + alpha_histories

names = [
    "mc ordinary ",
    "mc weighted",

] + ["TD({}), alpha: {}".format(td_n, alpha) for alpha in alphas ]

for histories in list_of_histories:
    rmses = evaluate_experiment(histories, v_mc)

    run_lengths = [run_lenght for run_lenght, rmse in rmses]
    rmses = [rmse for run_lenght, rmse in rmses]



    mean = np.mean(rmses, axis=0)
    std = np.std(rmses, axis=0)

    plt.plot(run_lengths[0], mean)
    plt.fill_between(run_lengths[0], mean + std, mean - std, alpha=0.5)

plt.title("Blackjack")
plt.xlabel("Number of Episodes")
plt.ylabel("RMSE")
plt.legend(names)
plt.show()
