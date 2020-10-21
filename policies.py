import numpy as np


class Policy(object):
    def get_probs(self, states, actions):
        raise NotImplementedError

    def sample_action(self, state):
        raise NotImplementedError


black_jack_actions = {
    "hit": 1,
    "stick": 0
}


class RandomPolicy(Policy):

    def __init__(self, actions):
        self.actions = actions

    def get_probs(self, states, actions):
        probs = np.full(len(states), 1. / len(self.actions))
        return probs

    def sample_action(self, state):
        probs = self.get_probs([state for i in range(len(self.actions))], self.actions)

        action = np.random.choice(self.actions, p=probs)
        return action


class SimpleBlackjackPolicy(Policy):
    """
    A simple BlackJack policy that sticks with 20 or 21 points and hits otherwise.
    """

    def get_probs(self, states, actions):
        """
        This method takes a list of states and a list of actions and returns a numpy array that contains a probability
        of perfoming action in given state for every corresponding state action pair.

        Args:
            states: a list of states.
            actions: a list of actions.

        Returns:
            Numpy array filled with probabilities (same length as states and actions)
        """

        probs = []

        for (player_card_sum, dealers_card, usable_ace), action in zip(states, actions):
            if player_card_sum in [20, 21]:
                if action == black_jack_actions["stick"]:
                    probs.append(1.0)
                else:
                    probs.append(0.0)
            else:
                if action == black_jack_actions["hit"]:
                    probs.append(1.0)
                else:
                    probs.append(0.0)

        return np.array(probs)

    def sample_action(self, state):
        """
        This method takes a state as input and returns an action sampled from this policy.

        Args:
            state: current state

        Returns:
            An action (int).
        """
        # YOUR CODE HERE
        actions = [*black_jack_actions.values()]

        probs = self.get_probs([state, state], actions)

        action = np.random.choice(actions, p=probs)
        return action


class EpsilonGreedyPolicy(Policy):
    def __init__(self, actions, Q, epsilon):
        self.actions = actions
        self.Q = Q
        self.epsilon = epsilon

    def get_probs(self, states, actions):
        probs = np.full(len(states), self.epsilon / len(self.actions))
        for i, (state, action) in enumerate(zip(states, actions)):
            if np.argmax(self.Q[state]) == action:
                probs[i] += 1 - self.epsilon

        return probs

    def sample_action(self, state):
        probs = self.get_probs([state for i in range(len(self.actions))], self.actions)
        action = np.random.choice(self.actions, p=probs)
        return action


### Creator functions

def create_epsilon_greedy_nchain_policy(n, epsilon):
    actions = [0, 1]
    Q = np.zeros((n, 2))

    for state in range(n):
        Q[state, 0] = 1
    policy = EpsilonGreedyPolicy(actions, Q, 0.001)
    return policy


def create_epsilon_greey_frozenlake_policy(epsilon):
    action_map = {
        0: "D",
        1: "R",
        2: "D",
        3: "L",
        4: "D",
        5: "U",
        6: "D",
        7: "U",
        8: "R",
        9: "D",
        10: "D",
        11: "U",
        12: "U",
        13: "R",
        14: "R",
        15: "D",

    }
    index = {"L": 0, "D": 1, "R": 2, "U": 3}

    Q = np.zeros((16, 4))

    for state, action in action_map.items():
        Q[state, index[action]] = 1
    actions = [0,1,2,3]
    policy = EpsilonGreedyPolicy(actions, Q, 0.000001)
    return policy
