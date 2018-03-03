import numpy as np
from .base_approx import BaseApproximator
import gym

import pdb

# TODO: documentaiton & comments

class AggregateApproximator(BaseApproximator):

    def __init__(self, step_size, bins, init_val):
        assert isinstance(bins, list) or isinstance(bins, tuple)
        super().__init__()

        self._step_size = step_size
        self._nb_dims = len(bins)
        self._bin_sizes = np.array(bins)
        self._init_val = init_val
        
        # actuall initialisation happens in set_state_action_spaces()
        # called by the agent

    def set_state_action_spaces(self, state_space, action_space):

        # These should be relaxed in the future,
        # possibly remove gym dependancy
        if not isinstance(state_space, gym.spaces.Box):
            raise ValueError('Only gym.spaces.Box state space supproted')
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError('Only gym.spaces.Discrete action space supported')

        if len(state_space.shape) != 1:
            raise ValueError('state_space.shape must be a 1D vector')

        if state_space.shape != self._bin_sizes.shape:
            raise ValueError('Input shape does not match state_space shape')

        
        self._state_space = state_space
        self._action_space = action_space

        eps = 1e-5

        self._bin_borders = []  # array of bin borders for each dimension
        for i in range(self._nb_dims):
            bin_size = self._bin_sizes[i]
            low = self._state_space.low[i]
            high = self._state_space.high[i]
            borders = np.linspace(low, high+eps, bin_size+1)
            self._bin_borders.append(borders)

        nb_actions = action_space.n
        self._states = \
            np.zeros([*self._bin_sizes, nb_actions]) + self._init_val


    def get_weights_fingerprint(self):
        return np.sum(self._states)

    def _to_idx(self, state):
        indices = []  # array of indices, one int per dimension
        for i in range(self._nb_dims):
            idx = np.digitize(state[i], self._bin_borders[i]) - 1
            indices.append(idx)
        return indices

    def estimate(self, state, action):
        assert self._state_space is not None
        assert self._action_space is not None
        assert self._state_space.contains(state)
        assert self._action_space.contains(action)

        indices = self._to_idx(state)

        return self._states[(*indices, action)]
    
    def estimate_all(self, states):
        assert self._state_space is not None
        assert self._action_space is not None
        assert all(map(self._state_space.contains, states))

        result = np.zeros( [len(states), self._action_space.n], dtype=float)
        for si in range(len(states)):
            for i in range(self._action_space.n):
                action = i
                result[si, i] = self.estimate(states[si], action)

        return result

    def train(self, states, actions, targets):

        #
        #   If single state/action/target was passed
        #
        if states.ndim == 1:
            assert self._state_space.contains(states)
            assert self._action_space.contains(actions)
            assert np.isscalar(targets)
            self._update(states, actions, targets)
            return

        #
        #   If arrays were passed
        #
        assert self._state_space is not None
        assert self._action_space is not None
        
        assert isinstance(states, np.ndarray) or isinstance(states, list)
        assert states.shape[1:] == self._state_space.shape
        # assert all(map(self._state_space.contains, states))

        assert isinstance(actions, np.ndarray) or isinstance(states, list)
        assert actions.shape[1:] == self._action_space.shape
        # assert all(map(self._action_space.contains, actions))

        assert isinstance(targets, np.ndarray) or isinstance(states, list)
        assert targets.ndim == 1

        assert len(states) == len(actions) == len(targets)

        for i in range(len(states)):
            self._update(states[i], actions[i], targets[i])


    def _update(self, state, action, target):

        est = self.estimate(state, action)

        indices = self._to_idx(state)
        act_idx = action
        
        self._states[(*indices, act_idx)] += \
            self._step_size * (target - est)

