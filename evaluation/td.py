from collections import defaultdict
from tqdm import tqdm
import numpy as np


def n_step_td(env, behavior_policy, alpha, num_episodes, sampling_function, n=1,
              discount_factor=1.0):
    V = defaultdict(float)

    # YOUR CODE HERE
    for i in tqdm(range(num_episodes)):
        current_state = env.reset()

        T = np.inf
        t = 0
        tau = - n
        rewards = []
        states = []
        actions = []

        while tau < T:
            if t < T:
                current_state, action, next_state, reward, done = sampling_function(env, behavior_policy,
                                                                                    current_state)
                states.append(current_state)
                actions.append(action)
                rewards.append(reward)

                if done:
                    T = t + 1

            if tau >= 0:

                G = np.sum([discount_factor ** i * rewards[i] for i in range(tau, min(tau + n, T))])

                if tau + n < T:
                    G = G + discount_factor ** n * V[states[tau + n]]

                V[states[tau]] = V[states[tau]] + alpha * (G - V[states[tau]])

            current_state = next_state
            tau = t - n + 1
            t += 1

    return V


def n_step_td_off_policy(env, behavior_policy, target_policy, num_episodes, sampling_function, n=1,
                            discount_factor=1.0, alpha=0.001):
    V = defaultdict(float)

    for i in tqdm(range(num_episodes)):
        current_state = env.reset()

        T = np.inf
        t = 0
        tau = - n
        rewards = []
        states = []
        actions = []

        while tau < T:
            if t < T:
                current_state, action, next_state, reward, done = sampling_function(env, behavior_policy,
                                                                                    current_state)
                states.append(current_state)
                actions.append(action)
                rewards.append(reward)

                if done:
                    T = t + 1

            if tau >= 0:
                tau_upto = min(tau + n - 1, T)

                ro = np.product(
                    [target_policy.get_probs([state], [action])[0] / behavior_policy.get_probs([state], [action])[0] for
                     state, action, reward in
                     zip(states[tau: tau_upto], actions[tau: tau_upto], rewards[tau: tau_upto])])

                G = np.sum([discount_factor ** i * rewards[i] for i in range(tau, min(tau + n, T))])
                if tau + n < T:
                    G = G + discount_factor ** n * V[states[tau + n]]

                V[states[tau]] = V[states[tau]] + alpha * ro *  (G - V[states[tau]])

            current_state = next_state
            tau = t - n + 1
            t += 1

    return V


