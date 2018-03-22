import numpy as np
import gym
import pdb

# TODO: Under Construction
# TODO: Rename to PolicyGradientContinous
# TODO: Change all NotImplemented to NotImplementedError
# TODO: wrapp all scalar states/actions everywehre into 0-dim np array?

class PolicyTabularCont:
    def __init__(self, learn_rate, std_dev):
        """Under construction"""
        self._learn_rate = learn_rate
        self._state_space = None
        self._action_space = None
        # self._v_approx = None
        # self._q_approx = None
        self._weights = None          # means for different states
        self._std_dev = std_dev
        self._variance = std_dev**2   # variance not parametrized

    def set_state_action_spaces(self, state_space, action_space):
        # These should be relaxed in the future,
        # possibly remove gym dependancy
        if not isinstance(state_space, gym.spaces.Discrete):
            raise ValueError('Only gym.spaces.Discrete state space supproted')
        if not isinstance(action_space, gym.spaces.Box):
            raise ValueError('Only gym.spaces.Box action space supported')
        if action_space.shape != ():
            raise ValueError('Only scalar actions supported for now')

        self._state_space = state_space
        self._action_space = action_space

        self._weights = np.zeros([state_space.n])


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
        # just sample normal distribution
        mean = self._weights[state]
        action = np.random.normal(mean, self._std_dev)
        return np.array(action)

    def train_single(self, state, action, target):

        features = np.zeros([self._state_space.n])
        features[state] = 1

        mean = self._weights[state]
        log_grad_st = (action - mean) * features
        log_grad_st /= self._variance**2

        delta_weights = self._learn_rate * target * log_grad_st
        self._weights += delta_weights

    def train_batch(self, states, actions, targets):
        assert self._state_space is not None
        assert self._action_space is not None

        assert isinstance(states, np.ndarray)
        assert states.shape[1:] == self._state_space.shape
        # assert all(map(self._state_space.contains, states))

        assert isinstance(actions, np.ndarray)
        assert actions.shape[1:] == self._action_space.shape
        # assert all(map(self._action_space.contains, actions))

        assert isinstance(targets, np.ndarray)
        assert targets.ndim == 1

        assert len(states) == len(actions) == len(targets)

        gradient = np.zeros([self._state_space.n])

        for i in range(len(states)):
            state = states[i]
            action = actions[i]
            target = targets[i]

            features = np.zeros([self._state_space.n])
            features[state] = 1

            mean = self._weights[state]
            log_grad_st = (action - mean) * features
            log_grad_st /= self._variance**2

            delta_weights = self._learn_rate * target * log_grad_st
            gradient += delta_weights

        self._weights += gradient


    def get_raw(self, state):
        """Regurns mean action for state"""
        return self._weights[state]