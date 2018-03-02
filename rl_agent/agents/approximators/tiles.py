import numpy as np
from . import tile_coding
import gym

import pdb

class TilesApproximator:

    def __init__(self, step_size, num_tillings, init_val):

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

        # print(mem_req)
        # pdb.set_trace()


        # while True:
        #     pos = np.random.uniform(-1.2, 0.6)
        #     vel = np.random.uniform(-0.07, 0.07)
        #     action = np.random.randint(0, 3)

        #     active_tiles = tile_coding.tiles(
        #         self._hashtable, self._num_tillings,
        #         [self._scales[0] * pos, self._scales[1] * vel], [action])

        #     print(len(self._hashtable.dictionary))


    def get_weights_fingerprint(self):
        return np.sum(self._weights)

    def _test_input(self, state, action):
        assert isinstance(state, np.ndarray)
        assert isinstance(state[0], float)
        assert isinstance(state[1], float)
        assert isinstance(action, int) or isinstance(action, np.int64)

        pos, vel = state[0], state[1]

        assert -1.2 <= pos and pos <= 0.5
        assert -0.07 <= vel and vel <= 0.07

        assert action in [0, 1, 2]

        return pos, vel, action

    def estimate(self, state, action):
        pos, vel, action = self._test_input(state, action)

        active_tiles = tile_coding.tiles(
            self._hashtable, self._num_tillings,
            [self._scales[0] * pos, self._scales[1] * vel], [action])

        return np.sum(self._weights[active_tiles])

    def estimate_all(self, states):
        assert isinstance(states, np.ndarray)
        assert states.ndim == 2
        assert len(states) > 0
        assert states.shape[1] == 2   # pos, vel
        assert states.dtype == np.float32 or states.dtype == np.float64
        assert np.min(states, axis=0)[0] >= -1.2  # pos
        assert np.max(states, axis=0)[0] <= 0.5  # pos
        assert np.min(states, axis=0)[1] >= -0.07  # vel
        assert np.max(states, axis=0)[1] <= 0.07  # vel

        result = np.zeros( [len(states), self._action_space.n], dtype=float)
        for si in range(len(states)):
            for i in range(self._action_space.n):
                action = i
                result[si, i] = self.estimate(states[si], action)

        return result


    def train(self, state, action, target):
        pos, vel, action = self._test_input(state, action)
        assert pos < 0.5  # this should never be called on terminal state

        active_tiles = tile_coding.tiles(
            self._hashtable, self._num_tillings,
            [self._scales[0] * pos, self._scales[1] * vel],
            [action])

        est = np.sum(self._weights[active_tiles])

        delta = self._step_size * (target - est)

        for tile in active_tiles:
            self._weights[tile] += delta

    def update2(self, states, actions, rewards_n, states_n, dones, timing_dict):

        # pdb.set_trace()
        # print('hop')

        est_arr = self.estimate_all(states)
        est_arr_1 = self.estimate_all(states_n)

        errors = np.zeros([len(states)])

        for i in range(len(states)):
            St = states[i]
            At = actions[i, 0]
            Rt_1 = rewards_n[i, 0]
            St_1 = states_n[i]
            done = dones[i, 0]

            est = est_arr_1[i]
            At_1 = _rand_argmax(est)

            if done:
                Tt = Rt_1
            else:
                Tt = Rt_1 + 0.99 * self.estimate(St_1, At_1)

            errors[i] = Tt - est_arr[i, At]

            self.update(St, At, Tt)

        return errors
