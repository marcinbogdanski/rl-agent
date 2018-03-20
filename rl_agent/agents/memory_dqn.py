import numpy as np
import gym
import pdb

# TODO: Add proper unittest

class MemoryDQN:
    """Circular buffer for DQN memory reply."""

    def __init__(self,
                 max_len,
                 batch_size,
                 enable_pmr=False,
                 initial_pmr_error=1000.0):
        """
        Args:
            max_len: maximum capacity
            enable_pmr: if True, enable Marcins version of PMR
            initial_pmr_error: error for new samples, should be order of
                magnitude larger than max error during normal operation
        """
        assert isinstance(max_len, int)
        assert max_len > 0

        self._max_len = max_len
        self._batch_size = batch_size
        self._enable_pmr = enable_pmr
        self._initial_pmr_error = initial_pmr_error

        self._curr_insert_ptr = 0
        self._curr_len = 0

    def set_state_action_spaces(self, state_space, action_space):
        """Set state/action space descriptors
        Params:
            state_space: gym.spaces.Box (tested) or Discrete (not tested)
            action_space: gym.spaces.Box (not tested) or Discrete (tested)
        """
            
        # These should be relaxed in the future to support more spaces,
        # possibly remove gym dependancy
        if not isinstance(state_space, gym.spaces.Box):
            raise ValueError('Only gym.spaces.Box state space supproted')
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError('Only gym.spaces.Discrete action space supported')

        assert state_space.shape is not None
        assert state_space.dtype is not None
        assert action_space.shape is not None
        assert action_space.dtype is not None

        self._state_space = state_space
        self._action_space = action_space

        

        St_shape = [self._max_len] + list(state_space.shape)
        At_shape = [self._max_len] + list(action_space.shape)
        Rt_1_shape = [self._max_len]
        St_1_shape = [self._max_len] + list(state_space.shape)
        done_shape = [self._max_len]
        error_shape = [self._max_len]

        self._hist_St = np.zeros(St_shape, dtype=state_space.dtype)
        self._hist_At = np.zeros(At_shape, dtype=action_space.dtype)
        self._hist_Rt_1 = np.zeros(Rt_1_shape, dtype=float)
        self._hist_St_1 = np.zeros(St_1_shape, dtype=state_space.dtype)
        self._hist_done = np.zeros(done_shape, dtype=bool)
        self._hist_error = np.zeros(error_shape, dtype=float)

        self._log_every = None
        self._log_mem = None


    def append(self, St, At, Rt_1, St_1, done):
        """Add one sample to memory, override oldest if max_len reached.

        Args:
            St - state
            At - action
            Rt_1 - reward
            St_1 - next state
            done - True if episode completed
        """
        assert self._state_space is not None
        assert self._action_space is not None
        assert self._state_space.contains(St)
        assert self._action_space.contains(At)
        assert self._state_space.contains(St_1)

        self._hist_St[self._curr_insert_ptr] = St
        self._hist_At[self._curr_insert_ptr] = At
        self._hist_Rt_1[self._curr_insert_ptr] = Rt_1
        self._hist_St_1[self._curr_insert_ptr] = St_1
        self._hist_done[self._curr_insert_ptr] = done
        
        # arbitrary high def error
        self._hist_error[self._curr_insert_ptr] = self._initial_pmr_error

        #
        #   increment insertion pointer, roll back if required 
        #
        if self._curr_len < self._max_len:
            self._curr_len += 1

        self._curr_insert_ptr += 1 
        if self._curr_insert_ptr >= self._max_len:
            self._curr_insert_ptr = 0

    def clear(self):
        self._curr_len = 0
        self._curr_insert_ptr = 0
        self._hist_St.clear()
        self._hist_At.clear()
        self._hist_Rt_1.clear()
        self._hist_St_1.clear()
        self._hist_done.clear()
        self._hist_error.clear()

    def length(self):
        """Number of samples in memory, 0 <= length <= max_len"""
        return self._curr_len

    def get_batch(self, batch_len=None):
        """Sample batch of data, with repetition

        Args:
            batch_len: nb of samples to pick,
                       defaults to value passed in constructor

        Returns:
            states, actions, rewards, next_states, done, indices
            Each returned element is np.ndarray with length == batch_len
            Last element 'indices' can be passed back update_errors() method
        """
        assert self._state_space is not None
        assert self._action_space is not None
        assert self._curr_len > 0
        assert batch_len is None or batch_len > 0

        if batch_len is None:
            batch_len = self._batch_size

        if not self._enable_pmr:
            # np.random.randint much faster than np.random.sample (?)
            indices = np.random.randint(
                low=0, high=self._curr_len, size=batch_len, dtype=int)

        else:
            cdf = np.cumsum(self._hist_error+0.01)
            cdf = cdf / cdf[-1]
            values = np.random.rand(batch_len)
            indices = np.searchsorted(cdf, values)


        states = np.take(self._hist_St, indices, axis=0)
        actions = np.take(self._hist_At, indices, axis=0)
        rewards_1 = np.take(self._hist_Rt_1, indices, axis=0)
        states_1 = np.take(self._hist_St_1, indices, axis=0)
        dones = np.take(self._hist_done, indices, axis=0)

        return states, actions, rewards_1, states_1, dones, indices

    def update_errors(self, indices, errors):
        """For PMR, update error values for specified indices.

        Example:
            memory = Memory(...)
            ... # add some data
            st, act, rew, st_1, done, indices = memory.get_batch(64)
            ... # train neural network
            ... # calculate error values for each element in batch
            ... # but do NOT modify memory in any way
            memory.update_errors(indices, np.abs(errors))
        """
        assert self._state_space is not None
        assert self._action_space is not None
        assert isinstance(indices, np.ndarray)
        assert indices.ndim == 1
        assert len(indices) > 0
        assert isinstance(errors, np.ndarray)
        assert errors.ndim == 1
        assert len(indices) == len(errors)
        assert (errors > 0).all

        self._hist_error[indices] = errors

    def install_logger(self, logger, log_every):
        self._log_mem = logger
        self._log_every = log_every

    def perform_logging(self, episode, step, total_step):

        if self._log_mem is not None and not self._log_mem.is_initialized:
            self._log_mem.add_param('max_len', self._max_len)
            self._log_mem.add_param('enable_pmr', self._enable_pmr)
            self._log_mem.add_data_item('curr_size')
            self._log_mem.add_data_item('hist_St')
            self._log_mem.add_data_item('hist_At')
            self._log_mem.add_data_item('hist_Rt_1')
            self._log_mem.add_data_item('hist_St_1')
            self._log_mem.add_data_item('hist_done')
            self._log_mem.add_data_item('hist_error')

        #
        #   Log Memory
        #
        if self._log_mem is not None and total_step % self._log_every == 0:

            ptr = self._curr_insert_ptr
            self._log_mem.append(
                episode, step, total_step,
                curr_size=self.length(),
                hist_St=np.concatenate((self._hist_St[ptr:], self._hist_St[0:ptr])),
                hist_At=np.concatenate((self._hist_At[ptr:], self._hist_At[0:ptr])),
                hist_Rt_1=np.concatenate((self._hist_Rt_1[ptr:], self._hist_Rt_1[0:ptr])),
                hist_St_1=np.concatenate((self._hist_St_1[ptr:], self._hist_St_1[0:ptr])),
                hist_done=np.concatenate((self._hist_done[ptr:], self._hist_done[0:ptr])),
                hist_error=np.concatenate((self._hist_error[ptr:], self._hist_error[0:ptr])) )
        
    def _print_all(self):
        print()
        print('_hist_St')
        print(self._hist_St)

        print()
        print('_hist_At')
        print(self._hist_At)

        print()
        print('_hist_Rt_1')
        print(self._hist_Rt_1)

        print()
        print('_hist_St_1')
        print(self._hist_St_1)

        print()
        print('_hist_done')
        print(self._hist_done)

if __name__ == '__main__':
    # this is old test-method

    mem = Memory(state_shape=(2, ),
                 act_shape=(1, ),
                 dtypes=(float, int, float, float, bool),
                 max_len=10)

    i = 1
    mem.append(St=np.array([i, i], dtype=float),
        At=i, Rt_1=-i, St_1=np.array([i+1,i+1], dtype=float), done=False)
    res = mem.get_batch(3)
    print(res)

    i = 2
    mem.append(St=np.array([i, i], dtype=float),
        At=i, Rt_1=-i, St_1=np.array([i+1,i+1], dtype=float), done=False)
    res = mem.get_batch(3)
    print(res)

    i = 3
    mem.append(St=np.array([i, i], dtype=float),
        At=i, Rt_1=-i, St_1=np.array([i+1,i+1], dtype=float), done=False)
    res = mem.get_batch(3)
    print(res)


    for i in range(4, 12):
        mem.append(St=np.array([i, i], dtype=float),
            At=i, Rt_1=-i, St_1=np.array([i+1,i+1], dtype=float), done=False)

    i = 12
    mem.append(St=np.array([i, i], dtype=float),
        At=i, Rt_1=-i, St_1=np.array([i+1,i+1], dtype=float), done=True)
    res = mem.get_batch(3)
    print(res)
