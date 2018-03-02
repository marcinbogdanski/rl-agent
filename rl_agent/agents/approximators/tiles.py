import numpy as np

from . import tile_coding

import pdb

class TilesApproximator:

    def __init__(self, step_size, action_space, init_val=0):
        self._num_of_tillings = 8
        self._step_size = step_size / self._num_of_tillings

        self._action_space = action_space

        self._pos_scale = self._num_of_tillings / (0.5 + 1.2)
        self._vel_scale = self._num_of_tillings / (0.07 + 0.07)

        self._hashtable = tile_coding.IHT(2048)
        self._weights = np.zeros(2048) + init_val / self._num_of_tillings
        
        max_len = 2000

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
            self._hashtable, self._num_of_tillings,
            [self._pos_scale * pos, self._vel_scale * vel], [action])

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

        result = np.zeros( [len(states), len(self._action_space)], dtype=float)
        for si in range(len(states)):
            for i in range(len(self._action_space)):
                action = self._action_space[i]
                result[si, i] = self.estimate(states[si], action)

        return result


    def update(self, state, action, target):
        pos, vel, action = self._test_input(state, action)
        assert pos < 0.5  # this should never be called on terminal state

        active_tiles = tile_coding.tiles(
            self._hashtable, self._num_of_tillings,
            [self._pos_scale * pos, self._vel_scale * vel],
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
