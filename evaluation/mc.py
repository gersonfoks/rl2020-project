from collections import defaultdict
from tqdm import tqdm



def mc_ordinary_importance_sampling(env, behavior_policy, target_policy, num_episodes, sampling_function,
                                    discount_factor=1.0):
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

    # YOUR CODE HERE
    for i in tqdm(range(num_episodes)):
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

    return V


def mc_weighted_importance_sampling(env, behavior_policy, target_policy, num_episodes, sampling_function,
                                    discount_factor=1.0):
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

    returns_count = defaultdict(float)

    # YOUR CODE HERE
    for i in tqdm(range(num_episodes)):
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

    return V
