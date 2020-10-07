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

class EpsilonGreedyPolicy(object):
    def __init__(self, actions, Q, epsilon):
        self.actions = actions
        self.Q = Q
        self.epsilon = epsilon

    def get_probs(self, states, actions):
        probs = np.full(len(states), self.epsilon/len(actions))    
        index = np.random.choice(np.flatnonzero(self.Q[states[0]] == self.Q[states[0]].max()))
        probs[index] += 1-self.epsilon
        return probs


    def sample_action(self, state):
        probs = self.get_probs([state for i in range(len(self.actions))], self.actions)
        action = np.random.choice(self.actions, p=probs)
        return action


    