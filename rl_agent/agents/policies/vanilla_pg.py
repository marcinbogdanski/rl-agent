import numpy as np
import gym
import pdb

# TODO: Under Construction
# TODO: Change all NotImplemented to NotImplementedError

def _softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

class VanillaPolicyGradient:
    def __init__(self, learn_rate):
        """Under construction"""
        self._learn_rate = learn_rate
        self._state_space = None
        self._action_space = None
        # self._v_approx = None
        # self._q_approx = None
        self._weights = None

    def set_state_action_spaces(self, state_space, action_space):
        # These should be relaxed in the future,
        # possibly remove gym dependancy
        if not isinstance(state_space, gym.spaces.Discrete):
            raise ValueError('Only gym.spaces.Discrete state space supproted')
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError('Only gym.spaces.Discrete action space supported')

        self._state_space = state_space
        self._action_space = action_space

        self._weights = np.zeros([state_space.n, action_space.n])


    def link(self, agent):
        pass
        # if hasattr(agent, 'V'):
        #     self._v_approx = agent.V
        # if hasattr(agent, 'Q'):
        #     self._q_approx = agent.Q


    def reset(self):
        """Reset at the end of episode"""
        pass

    def next_step(self, total_step):
        pass

    def pick_action(self, state):

        # Fully generic would look like this:
        # features = np.zeros(self._state_space.n)
        # features[state] = 1
        # preferences = np.dot(features, self._weights)
        # assert (preferences == self._weights[state]).all()

        pref_all_act = self._weights[state]
        prob_all_act = _softmax(pref_all_act)

        act = np.random.choice(range(len(prob_all_act)), p=prob_all_act)
        return act

    def train(self, states, actions, targets):
        self._update(states, actions, targets)

    def _update(self, state, action, target):

        features = np.zeros([self._state_space.n, self._action_space.n])
        features[state, action] = 1

        pref_all_act = self._weights[state]
        prob_all_act = _softmax(pref_all_act)

        # Sutton & Barto eq 13.7
        log_grad_st = np.copy(features)
        log_grad_st[state] -= prob_all_act

        self._weights += self._learn_rate * target * log_grad_st

        pass
        #raise NotImplementedError

    def get_raw(self, state):
        """Regurns probability distribuition for actions"""

        pref_all_act = self._weights[state]
        prob_all_act = _softmax(pref_all_act)

        return prob_all_act