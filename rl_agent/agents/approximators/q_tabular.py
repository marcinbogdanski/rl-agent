import numpy as np
from .approx_base import ApproximatorBase
import gym

import pdb

# TODO: documentaiton & comments

class QFunctTabular(ApproximatorBase):

    def __init__(self, step_size, init_val):
        """Tabular state-action function

        See base class documentation for method descriptions
        
        Args:
            step_size: learning rate
            init_val: initial value for the table
        """
        super().__init__()

        self._step_size = step_size
        self._init_val = init_val
        

    def set_state_action_spaces(self, state_space, action_space):

        # These should be relaxed in the future,
        # possibly remove gym dependancy
        if not isinstance(state_space, gym.spaces.Discrete):
            raise ValueError('Only gym.spaces.Discrete state space supproted')
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError('Only gym.spaces.Discrete action space supported')
        
        self._state_space = state_space
        self._action_space = action_space

        self._weights = np.zeros([state_space.n, action_space.n]) \
                         + self._init_val


    def get_weights_fingerprint(self):
        return np.sum(self._weights)


    def estimate(self, state, action):
        assert self._state_space is not None
        assert self._action_space is not None
        assert self._state_space.contains(state)
        assert self._action_space.contains(action)

        return self._weights[state, action]
    
    def estimate_all(self, states):
        assert self._state_space is not None
        assert self._action_space is not None
        # assert all(map(self._state_space.contains, states))

        result = np.zeros( [len(states), self._action_space.n], dtype=float)
        for si in range(len(states)):
            for i in range(self._action_space.n):
                action = i
                result[si, i] = self.estimate(states[si], action)

        return result

    def train(self, states, actions, targets):
        assert self._state_space is not None
        assert self._action_space is not None

        #
        #   If single state/action/target was passed
        #
        if states.ndim == 0 or states.ndim == 1:
            assert self._state_space.contains(states)
            assert self._action_space.contains(actions)
            assert np.isscalar(targets)
            self._update(states, actions, targets)
            return

        #
        #   If arrays were passed
        #        
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
        est = self._weights[state, action]
        
        self._weights[state, action] += \
            self._step_size * (target - est)

