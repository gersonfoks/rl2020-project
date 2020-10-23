from collections import defaultdict

import numpy as np
from ast import literal_eval
import matplotlib.pyplot as plt
from utils.experiments import evaluate_experiment


def sample_episode(env, policy):
    """
    A sampling routine. Given environment and a policy samples one episode and returns states, actions, rewards
    and dones from environment's step function and policy's sample_action function as lists.

    Args:
        env: OpenAI gym environment.
        policy: A policy which allows us to sample actions with its sample_action method.

    Returns:
        Tuple of lists (states, actions, rewards, dones). All lists should have same length.
        Hint: Do not include the state after the termination in the list of states.
    """
    states = []
    actions = []
    rewards = []
    dones = []

    state = env.reset()

    while True:
        states.append(state)

        action = policy.sample_action(state)
        state, reward, done, _ = env.step(action)

        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        if done == True:
            break

    return states, actions, rewards, dones


def sample_step(env, policy, current_state):
    """
    A sampling routine. Given environment and a policy samples one episode and returns states, actions, rewards
    and dones from environment's step function and policy's sample_action function as lists.

    Args:
        env: OpenAI gym environment.
        policy: A policy which allows us to sample actions with its sample_action method.

    Returns:
        Tuple of lists (states, actions, rewards, dones). All lists should have same length.
        Hint: Do not include the state after the termination in the list of states.
    """

    action = policy.sample_action(current_state)
    next_state, reward, done, _ = env.step(action)

    return current_state, action, next_state, reward, done


def save_v_history(V_hist, name):
    v_history = {key: dict(value) for key, value in V_hist.items()}
    with open("{}.txt".format(name), 'w') as outfile:
        outfile.write(str(v_history))


def load_v_history(name):
    with open("{}.txt".format(name), "r") as file:
        v_history = literal_eval(file.read())


    ## Transform back to default dict
    v_history = {key: defaultdict(float, value) for key, value in v_history.items()}
    return v_history


def get_predicted_values(V, player_values, dealer_values, usable_ace):
    predicted_values = np.zeros([len(dealer_values), len(player_values)])
    for player_value in player_values:
        for dealer_value in dealer_values:
            predicted_values[dealer_value - 1, player_value - 12] = V[(player_value, dealer_value, usable_ace)]
    return predicted_values


def sample_episode_lim(env, policy, limit=100):
    states = []
    actions = []
    rewards = []
    dones = []

    state = env.reset()

    for i in range(limit):
        states.append(state)

        action = policy.sample_action(state)
        state, reward, done, _ = env.step(action)

        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        if done == True:
            break

    return states, actions, rewards, dones


def get_oldest_history(histories):

    max_iter = np.max([key for key in histories.keys()])

    return histories[max_iter]



def plot_rmse_histories(list_of_histories, baseline, names):
    for histories in list_of_histories:
        rmses = evaluate_experiment(histories, baseline)

        run_lengths = [run_lenght for run_lenght, rmse in rmses]
        rmses = [rmse for run_lenght, rmse in rmses]

        mean = np.mean(rmses, axis=0)
        std = np.std(rmses, axis=0)

        plt.plot(run_lengths[0], mean)
        plt.fill_between(run_lengths[0], mean + std, mean - std, alpha=0.5)

    plt.legend(names)
    plt.show()