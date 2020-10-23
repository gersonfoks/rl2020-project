"""
This file contains the blackjack experiments.
"""
import gym
from environments.Nchain import NChainEnv
from evaluation.mc import *
from evaluation.td import n_step_td_off_policy
from utils.experiments import *
from utils.misc import sample_episode, sample_step
from policies import *
import matplotlib.pyplot as plt


from gym.envs.registration import register
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)



# Default variables

alphas = [0.01]
td_n = 4
actions = [0, 1,2,3]

env = gym.make('FrozenLakeNotSlippery-v0')

# Global settings


epsilon = 0.01

target_policy = create_epsilon_greey_frozenlake_policy(epsilon)
behavior_policy = RandomPolicy(actions)

n_experiments = 10

save_every = 1e2  ### How often we should save the results

# Conf for mc
n_mc_run = int(1e5)
save_every_mc = n_mc_run

# Conf for the mc off policy

n_mc_off_policy = int(1e5)

### Here we create the names
name = "frozenlake_small"


# First we need to run mc.
v_mc, hist = mc_prediction(env, target_policy, n_mc_run, sample_episode, save_every=n_mc_run, name="mc_{}".format(name))

histories_ord = run_experiments(mc_ordinary_importance_sampling, env, behavior_policy, target_policy, n_mc_off_policy,
                                sample_episode, n_experiments, save_every, name="mc_ord_{}".format(name))

histories_weighted = run_experiments(mc_weighted_importance_sampling, env, behavior_policy, target_policy,
                                     n_mc_off_policy, sample_episode, n_experiments, save_every,
                                     name="mc_weighted_{}".format(name))


alpha_histories = [

]
for alpha in alphas:
    alpha_histories.append(run_experiments(n_step_td_off_policy, env, behavior_policy, target_policy, n_mc_off_policy, sample_step,
                               n_experiments, save_every, name="td({})_{}_{}".format(td_n, name, alpha), alpha=alpha, n=td_n))

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

plt.title("Frozenlake")
plt.xlabel("Number of Episodes")
plt.ylabel("RMSE")
plt.legend(names)
plt.show()
