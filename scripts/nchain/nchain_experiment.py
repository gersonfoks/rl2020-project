"""
This file contains the blackjack experiments.
"""
import gym
from environments.Nchain import NChainEnv
from evaluation.mc import *
from evaluation.td import n_step_td_off_policy
from utils.experiments import *
from utils.misc import sample_episode, sample_step
from policies import SimpleBlackjackPolicy, RandomPolicy, create_epsilon_greedy_nchain_policy
import matplotlib.pyplot as plt

# Default variables

alphas = [0.1, 0.01, 0.001]

actions = [0, 1]
n = 4
env =  NChainEnv(n=n, slip=0.0)

# Global settings


epsilon = 0.01

target_policy = create_epsilon_greedy_nchain_policy(n, 0.001)
behavior_policy = create_epsilon_greedy_nchain_policy(n, 0.1)

n_experiments = 1

save_every = 1e2  ### How often we should save the results

# Conf for mc
n_mc_run = int(5e4)
save_every_mc = n_mc_run

# Conf for the mc off policy

n_mc_off_policy = int(5e4)

### Here we create the names
name = "{}Chain".format(n)


# First we need to run mc.
v_mc, hist = mc_prediction(env, target_policy, n_mc_run, sample_episode, save_every=n_mc_run, name="mc_{}".format(name))

histories_ord = run_experiments(mc_ordinary_importance_sampling, env, behavior_policy, target_policy, n_mc_off_policy,
                                sample_episode, n_experiments, save_every, name="mc_ord_".format(name))

histories_weighted = run_experiments(mc_weighted_importance_sampling, env, behavior_policy, target_policy,
                                     n_mc_off_policy, sample_episode, n_experiments, save_every,
                                     name="mc_weighted_".format(name))


alpha_histories = [

]
for alpha in alphas:
    alpha_histories.append(run_experiments(n_step_td_off_policy, env, behavior_policy, target_policy, n_mc_off_policy, sample_step,
                               n_experiments, save_every, name="td_{}_{}".format(name, alpha), alpha=alpha))

# Next we plot the results.

list_of_histories = [
    histories_ord,
    histories_weighted,

] + alpha_histories


names = [
    "mc ordinary ",
    "mc weighted",

] + ["TD, alpha: {}".format(alpha) for alpha in alphas ]

for histories in list_of_histories:
    rmses = evaluate_experiment(histories, v_mc)

    run_lengths = [run_lenght for run_lenght, rmse in rmses]
    rmses = [rmse for run_lenght, rmse in rmses]



    mean = np.mean(rmses, axis=0)
    std = np.std(rmses, axis=0)

    plt.plot(run_lengths[0], mean)
    plt.fill_between(run_lengths[0], mean + std, mean - std, alpha=0.5)


plt.legend(names)
plt.show()
