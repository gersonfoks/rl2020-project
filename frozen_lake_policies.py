from policies import Policy
import numpy as np

class FrozenLakeSmallPolicy(Policy):

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

