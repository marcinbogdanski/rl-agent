import numpy as np
from ...util import tile_coding
from .approx_base import ApproximatorBase
import gym

import pdb

class QFunctTiles(ApproximatorBase):
    def __init__(self, step_size, num_tillings, init_val):
        """Use tilings as state features with linear approximator

        See Sutton and Barto 2018 chap 9.5 Feature Construction for details

        See base class documentation for method descriptions
        
        Args:
            step_size: learning rate
            num_tillings: must be power of 2,
                must be greater or equal to four times num dimensions
            init_val: initial value for tilings, should be higher than
                maximum possible return to enable optimistic exploration
        """
        super().__init__()

        def is_power_of_2(num):
            return num != 0 and ((num & (num - 1)) == 0)

        if not is_power_of_2(num_tillings):
            raise ValueError('num_tillings must be power of 2')

        self._step_size = step_size / num_tillings
        self._num_tillings = num_tillings
        self._init_val = init_val


    def set_state_action_spaces(self, state_space, action_space):

        # These should be relaxed in the future,
        # possibly remove gym dependancy
        if not isinstance(state_space, gym.spaces.Box):
            raise ValueError('Only gym.spaces.Box state space supproted')
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError('Only gym.spaces.Discrete action space supported')

        if len(state_space.shape) != 1:
            raise ValueError('state_space.shape must be a 1D vector')

        num_dim = state_space.shape[0]
        if self._num_tillings < 4 * num_dim:
            # required by tile_coding code
            raise ValueError(
                'num_tilings must be >= four times state_space dimensions')

        self._state_space = state_space
        self._action_space = action_space

        self._scales = self._num_tillings / (state_space.high - state_space.low)

        #
        #   Memory required to express whole state/action space
        #
        #   E.g. 8 tillings, 2x state dimensions, 3x actions
        #   possible states:  (8+1)*(8+1)*(8) == 648
        #                     (dim_1_tiles + 1, dim_2_tiles + 1, nb_tiles)
        #   we need to add +1, becouse tiles go outside state space
        #   total possible states: possible states * nb_actions
        #                          648 * 3 = 1944
        #   note that actual memory requirement is slightly lower
        #   due to how tiles behave at the edges on n-dimensional qube
        #
        mem_req = (self._num_tillings+1)**num_dim * self._num_tillings
        mem_req *= self._action_space.n

        self._hashtable = tile_coding.IHT(mem_req)
        self._weights = np.zeros(mem_req) + self._init_val / self._num_tillings


    def get_weights_fingerprint(self):
        return np.sum(self._weights)


    def estimate(self, state, action):
        assert self._state_space is not None
        assert self._action_space is not None
        assert self._state_space.contains(state)
        assert self._action_space.contains(action)

        scaled_state = np.multiply(self._scales, state)

        active_tiles = tile_coding.tiles(
            self._hashtable, self._num_tillings,
            scaled_state, [action])

        return np.sum(self._weights[active_tiles])


    def estimate_all(self, states):
        assert self._state_space is not None
        assert self._action_space is not None
        assert isinstance(states, np.ndarray)
        assert states.shape[1:] == self._state_space.shape
        # assert all(map(self._state_space.contains, states))

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

        scaled_state = np.multiply(self._scales, state)

        active_tiles = tile_coding.tiles(
            self._hashtable, self._num_tillings,
            scaled_state, [action])

        est = np.sum(self._weights[active_tiles])

        delta = self._step_size * (target - est)

        self._weights[active_tiles] += delta

    