import numpy as np

import gym

import pdb



class AggregateApproximator:

    def __init__(self, step_size, bins, init_val):
        assert isinstance(bins, list) or isinstance(bins, tuple)

        self._step_size = step_size
        self._bins = np.array(bins)
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

        if state_space.shape != self._bins.shape:
            raise ValueError('Input shape does not match state_space shape')

        
        
        self._state_space = state_space
        self._action_space = action_space

        

        assert len(self._bins) == 2
        eps = 1e-5


        pos_bin_nb = self._bins[0]
        self._pos_bins = np.linspace(-1.2, 0.5+eps, pos_bin_nb+1)

        vel_bin_nb = self._bins[1]
        self._vel_bins = np.linspace(-0.07, 0.07+eps, vel_bin_nb+1)

        nb_actions = action_space.n
        self._states = np.zeros([pos_bin_nb, vel_bin_nb, nb_actions]) + self._init_val






    def get_weights_fingerprint(self):
        return np.sum(self._states)

    def _to_idx(self, state, action):
        assert self._state_space.contains(state)
        assert self._action_space.contains(action)

        pos, vel = state[0], state[1]
        act_idx = action

        pos_idx = np.digitize(pos, self._pos_bins) - 1
        if vel == 0.07:
            vel_idx = self._vel_bin_nb-1
        else:
            vel_idx = np.digitize(vel, self._vel_bins) - 1

        assert 0 <= pos_idx and pos_idx <= len(self._pos_bins)
        assert 0 <= vel_idx and vel_idx <= len(self._vel_bins)

        return pos_idx, vel_idx, act_idx

    def estimate(self, state, action):
        pos_idx, vel_idx, act_idx = self._to_idx(state, action)
        return self._states[pos_idx, vel_idx, act_idx]
    
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

    def update(self, state, action, target):
        pos_idx, vel_idx, act_idx = self._to_idx(state, action)

        pos = state[0]
        assert pos < 0.5  # this should never be called on terminal state
        
        est = self.estimate(state, action)
        
        self._states[pos_idx, vel_idx, act_idx] += \
            self._step_size * (target - est)

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