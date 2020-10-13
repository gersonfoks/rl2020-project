from collections import defaultdict
from tqdm import tqdm

from utils.misc import save_v_history


def mc_prediction(env, policy, num_episodes, sampling_function, discount_factor=1.0, save_every=-1, name="mc"):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.

    """

    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    V_hist = {}
    returns_count = defaultdict(float)
    # YOUR CODE HERE
    for i in tqdm(range(1, num_episodes+1)):
        G = 0

        states, actions, rewards, dones = sampling_function(env, policy)

        T = len(states)
        for t in range(T - 1, -1, -1):
            reward = rewards[t]
            G = discount_factor * G + reward
            state = states[t]

            returns_count[state] += 1
            V[state] = V[state] + (G - V[state]) / returns_count[state]

        if save_every > 0:
            if i % save_every == 0:
                V_hist[i] = V.copy()
                save_v_history(V_hist, name)


    return V, V_hist


def mc_ordinary_importance_sampling(env, behavior_policy, target_policy, num_episodes, sampling_function,
                                    discount_factor=1.0, save_every=-1, name="mc_ordinary"):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.

    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.

    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    returns_count = defaultdict(float)
    V_hist = {}
    # YOUR CODE HERE
    for i in tqdm(range(1, num_episodes+1)):
        G = 0

        states, actions, rewards, dones = sampling_function(env, behavior_policy)
        W = 1

        T = len(states)
        for t in range(T - 1, -1, -1):
            state = states[t]
            action = actions[t]
            reward = rewards[t]

            returns_count[state] += 1
            G = discount_factor * G + reward

            W = W * target_policy.get_probs([state], [action])[0] / behavior_policy.get_probs([state], [action])[0]

            # Update formula
            V[state] = V[state] + (W * G - V[state]) / returns_count[state]
        if save_every > 0:
            if i % save_every == 0:
                V_hist[i] = V.copy()
                save_v_history(V_hist, name)


    return V, V_hist


def mc_weighted_importance_sampling(env, behavior_policy, target_policy, num_episodes, sampling_function,
                                    discount_factor=1.0, save_every=-1, name="mc_weighted"):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and weighted importance sampling.

    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.

    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)

    C = defaultdict(float)
    V_hist = {}
    returns_count = defaultdict(float)

    # YOUR CODE HERE
    for i in tqdm(range(1, num_episodes+1)):
        G = 0

        states, actions, rewards, dones = sampling_function(env, behavior_policy)
        W = 1

        T = len(states)
        for t in range(T - 1, -1, -1):
            state = states[t]
            action = actions[t]
            reward = rewards[t]

            returns_count[state] += 1
            G = discount_factor * G + reward
            C[(state, action)] = C[(state, action)] + W
            W = W * target_policy.get_probs([state], [action])[0] / behavior_policy.get_probs([state], [action])[0]

            # Update formula
            V[state] = V[state] + W * (G - V[state]) / C[(state, action)]

            if W == 0:
                break
        if save_every > 0:
            if i % save_every == 0:
                V_hist[i] = V.copy()
                save_v_history(V_hist, name)
    return V, V_hist
