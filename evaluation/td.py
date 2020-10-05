from collections import defaultdict
from tqdm import tqdm
import numpy as np

def td_zero_importance_sampling(env, behavior_policy, target_policy, num_episodes, sampling_function,
                                discount_factor=1.0):
    V = defaultdict(float)

    returns_count = defaultdict(float)

    # YOUR CODE HERE
    for i in tqdm(range(num_episodes)):
        G = 0
        current_state = env.reset()

        while True:
            current_state, action, next_state, reward, done = sampling_function(env, behavior_policy,
                                                                                current_state)

            returns_count[current_state] += 1
            G = discount_factor * G + reward
            W = target_policy.get_probs([current_state], [action])[0] / \
                behavior_policy.get_probs([current_state], [action])[0]

            # Update formula
            V[current_state] = V[current_state] + W * (G - V[current_state]) / returns_count[current_state]

            current_state = next_state

            if done:
                break

    return V


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


def n_step_sarse_off_policy(env, behavior_policy, target_policy, num_episodes, sampling_function, n=1,
                            discount_factor=1.0):
    V = defaultdict(float)
    returns_count = defaultdict(float)
    alpha = 0.01
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
            returns_count[current_state] += 1
            if t < T:
                current_state, action, next_state, reward, done = sampling_function(env, behavior_policy,
                                                                                    current_state)
                states.append(current_state)
                actions.append(action)
                rewards.append(reward)

                if done:
                    T = t + 1

            if tau >= 0:
                tau_upto = min(tau + n, T)

                ro = np.product(
                    [target_policy.get_probs([state], [action])[0] / behavior_policy.get_probs([state], [action])[0] for
                     state, action, reward in
                     zip(states[tau: tau_upto], actions[tau: tau_upto], rewards[tau: tau_upto])])

                G = np.sum([discount_factor ** i * rewards[i] for i in range(tau, min(tau + n, T))])
                if tau + n < T:
                    G = G + discount_factor ** n * V[states[tau + n]]

                V[states[tau]] = V[states[tau]] + alpha * ro * (G - V[states[tau]])

            current_state = next_state
            tau = t - n + 1
            t += 1

    return V


def n_step_td_importance_sampling_test(env, behavior_policy, target_policy, num_episodes, sampling_function, n=1,
                                       discount_factor=1.0):
    V = defaultdict(float)

    alpha = 0.01
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
                tau_upto = min(tau + n, T)

                ro_s = [target_policy.get_probs([state], [action])[0] / behavior_policy.get_probs([state], [action])[0]
                        for
                        state, action, reward in
                        zip(states[tau: tau_upto], actions[tau: tau_upto], rewards[tau: tau_upto])]
                ro_prods = np.cumprod(ro_s)
                G = np.sum([ro_prods[i - tau] * discount_factor ** i * rewards[i] for i in range(tau, tau_upto)])

                mean = np.mean(ro_prods)

                G = G / mean

                if tau + n < T:
                    G = G + ro_prods[-1] * discount_factor ** n * V[states[tau + n]]

                if mean != 0:
                    V[states[tau]] = V[states[tau]] + alpha * (G - V[states[tau]])

            current_state = next_state
            tau = t - n + 1
            t += 1

    return V


def n_step_td_ordinary_importance_sampling(env, behavior_policy, target_policy, num_episodes, sampling_function, n=4,
                                           discount_factor=1.0):
    V = defaultdict(float)

    alpha = 0.01
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
                tau_upto = min(tau + n, T)

                ro_s = [target_policy.get_probs([state], [action])[0] / behavior_policy.get_probs([state], [action])[0]
                        for
                        state, action, reward in
                        zip(states[tau: tau_upto], actions[tau: tau_upto], rewards[tau: tau_upto])]
                ro_prods = np.cumprod(ro_s)
                G = np.sum([ro_prods[i - tau] * discount_factor ** i * rewards[i] for i in range(tau, tau_upto)])

                mean = len(ro_prods)

                G = G / mean

                if tau + n < T:
                    G = G + ro_prods[-1] * discount_factor ** n * V[states[tau + n]]

                if mean != 0:
                    V[states[tau]] = V[states[tau]] + alpha * (G - V[states[tau]])

            current_state = next_state
            tau = t - n + 1
            t += 1

    return V
