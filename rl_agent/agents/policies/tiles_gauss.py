import numpy as np
from ...util import tile_coding
import gym

# TODO: add support for parametrized variance
# TODO: add support for multiple continous actions
# TODO: add support for action scaling?

class TilesGaussPolicy:
    def __init__(self, step_size, num_tillings, std_dev):
        """Under construction"""

        def is_power_of_2(num):
            return num != 0 and ((num & (num - 1)) == 0)

        if not is_power_of_2(num_tillings):
            raise ValueError('num_tillings must be power of 2')

        self._step_size = step_size
        self._num_tillings = num_tillings
        self._std_dev = std_dev
        self._state_space = None
        self._action_space = None
        self._v_approx = None
        self._q_approx = None
        self._weights = None

    def set_state_action_spaces(self, state_space, action_space):
        # These should be relaxed in the future,
        # possibly remove gym dependancy
        if not isinstance(state_space, gym.spaces.Box):
            raise ValueError('Only gym.spaces.Box state space supproted')
        if not isinstance(action_space, gym.spaces.Box):
            raise ValueError('Only gym.spaces.Box action space supported')
        if action_space.shape != (1,):
            raise ValueError('Only single output action is supported for now')

        num_dim = state_space.shape[0]
        if self._num_tillings < 4 * num_dim:
            # required by tile_coding code
            raise ValueError(
                'num_tilings must be >= four times state_space dimensions')

        self._state_space = state_space
        self._action_space = action_space

        self._scales = self._num_tillings / (state_space.high - state_space.low)

        mem_req = (self._num_tillings+1)**num_dim * self._num_tillings
        
        self._hashtable = tile_coding.IHT(mem_req)
        self._weights = np.zeros(mem_req)

    def get_weights_fingerprint(self):
        return np.sum(self._weights)

    def link(self, agent):
        if hasattr(agent, 'V'):
            self._v_approx = agent.V
        if hasattr(agent, 'Q'):
            self._q_approx = agent.Q

    def sample_action(self, state):
        assert self._state_space is not None
        assert self._action_space is not None
        assert self._state_space.contains(state)

        scaled_state = np.multiply(self._scales, state)

        active_tiles = tile_coding.tiles(
            self._hashtable, self._num_tillings,
            scaled_state)

        gauss_mean = np.sum(self._weights[active_tiles])


    def reset(self):
        """Reset at the end of episode"""
        pass

    def next_step(self, total_step):
        pass

    def pick_action(self, state):
        raise NotImplemented()

    def train(self):
        raise NotImplemented