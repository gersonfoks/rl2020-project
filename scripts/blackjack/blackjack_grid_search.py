"""
This file does a gridsearch for TD for the frozenlake.

When we run it we get n=4, alpha =0.01
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

alphas = [0.1, 0.01, 0.001]

ns = [1, 2, 4, 8]

actions = [0,1]

env = gym.make('Blackjack-v0')


# Global settings


epsilon = 0.01

target_policy = SimpleBlackjackPolicy()
behavior_policy = RandomPolicy(actions)


n_experiments = 2

save_every = int(1e5)  ### How often we should save the results

# Conf for mc
n_mc_run = int(1e5)
save_every_mc = n_mc_run

# Conf for the mc off policy

n_mc_off_policy = int(1e5)

### Here we create the names
name = "frozenlake_small"


# First we need to run mc.
v_mc, hist = mc_prediction(env, target_policy, n_mc_run, sample_episode, save_every=n_mc_run, name="mc_{}".format(name))



histories = [

]
for alpha in alphas:
    for n in ns:


        histories.append(run_experiments(n_step_td_off_policy, env, behavior_policy, target_policy, n_mc_off_policy, sample_step,
                               n_experiments, save_every, name="td_{}_{}".format(name, alpha), alpha=alpha, n=n))

# Next we plot the results.




names = []
for alpha in alphas:
    for n in ns:
        names.append("alpha: {}, n: {}".format(alpha, n))

means = []
for histories in histories:
    rmses = evaluate_experiment(histories, v_mc)

    run_lengths = [run_lenght for run_lenght, rmse in rmses]
    rmses = [rmse for run_lenght, rmse in rmses]



    means.append(np.mean(rmses, axis=0)[-1])

print(means)
min_index = np.argmin(means)
print(np.min(means))
print(names[min_index])


