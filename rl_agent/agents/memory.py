import numpy as np
import pdb

class Memory:
    def __init__(self,
                 state_space,
                 action_space, 
                 max_len,
                 enable_pmr=False,
                 initial_pmr_error=1000.0):
        assert state_space.shape is not None
        assert state_space.dtype is not None
        assert action_space.shape is not None
        assert action_space.shape is not None
        assert isinstance(max_len, int)
        assert max_len > 0

        self._state_space = state_space
        self._action_space = action_space

        self._max_len = max_len
        self._enable_pmr = enable_pmr
        self._initial_pmr_error = initial_pmr_error
        self._curr_insert_ptr = 0
        self._curr_len = 0

        St_shape = [max_len] + list(state_space.shape)
        At_shape = [max_len] + list(action_space.shape)
        Rt_1_shape = [max_len]
        St_1_shape = [max_len] + list(state_space.shape)
        done_shape = [max_len]
        error_shape = [max_len]

        self._hist_St = np.zeros(St_shape, dtype=state_space.dtype)
        self._hist_At = np.zeros(At_shape, dtype=action_space.dtype)
        self._hist_Rt_1 = np.zeros(Rt_1_shape, dtype=float)
        self._hist_St_1 = np.zeros(St_1_shape, dtype=state_space.dtype)
        self._hist_done = np.zeros(done_shape, dtype=bool)
        self._hist_error = np.zeros(error_shape, dtype=float)

    def append(self, St, At, Rt_1, St_1, done):
        assert isinstance(St, np.ndarray)
        assert St.shape == self._state_space.shape
        assert St.dtype == self._hist_St.dtype
        assert self._state_space.contains(St)
        assert isinstance(At, int) or isinstance(At, np.int64)
        assert isinstance(Rt_1, float)
        assert isinstance(St_1, np.ndarray)
        assert St_1.shape == self._state_space.shape
        assert St_1.dtype == self._hist_St_1.dtype
        assert isinstance(done, bool)

        self._hist_St[self._curr_insert_ptr] = St
        self._hist_At[self._curr_insert_ptr] = At
        self._hist_Rt_1[self._curr_insert_ptr] = Rt_1
        self._hist_St_1[self._curr_insert_ptr] = St_1
        self._hist_done[self._curr_insert_ptr] = done
        # arbitrary high def error
        self._hist_error[self._curr_insert_ptr] = self._initial_pmr_error




        # if self._curr_len == self._max_len:
        #     np.savez('memory_3.npz',
        #         states=self._hist_St,
        #         actions=self._hist_At,
        #         rewards_1=self._hist_Rt_1,
        #         states_1=self._hist_St_1,
        #         dones=self._hist_done,
        #         )
        #     pdb.set_trace()




        # if self._curr_len == self._max_len:
        #     import matplotlib.pyplot as plt
        #     import log_viewer

        #     fig = plt.figure()
        #     ax = fig.add_subplot(111)

        #     indices = np.where(self._hist_done[:,0])[0]
            
        #     x_arr = self._hist_St[:,0]
        #     y_arr = self._hist_St[:,1]
        #     act_arr = self._hist_At[:,0]
        #     extent = (-1.2, 0.5, -0.07, 0.07)

        #     # plot full history
        #     # log_viewer.plot_trajectory_2d(
        #     #     ax, x_arr, y_arr, act_arr, extent, 0, -0.5)
            
        #     for idx in indices:
        #         log_viewer.plot_trajectory_2d(
        #             ax,
        #             x_arr[idx-100:idx+1],
        #             y_arr[idx-100:idx+1],
        #             act_arr[idx-100:idx+1], extent, 0, -0.5)

        #     plt.pause(0.1)
        #     pdb.set_trace()




        # if done:
        #     # done samples are very important
        #     # so make sure we multiply them a bit
        #     idx = np.random.randint(0, self._curr_len, size=5)
        #     self._hist_St[idx] = St
        #     self._hist_At[idx] = At
        #     self._hist_Rt_1[idx] = Rt_1
        #     self._hist_St_1[idx] = St_1
        #     self._hist_done[idx] = done

        if self._curr_len < self._max_len:
            self._curr_len += 1

        self._curr_insert_ptr += 1 
        if self._curr_insert_ptr >= self._max_len:
            self._curr_insert_ptr = 0

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

    def length(self):
        return self._curr_len

    def get_batch(self, batch_len):
        assert self._curr_len > 0
        assert batch_len > 0

        if not self._enable_pmr:
            indices = np.random.randint(
                low=0, high=self._curr_len, size=batch_len, dtype=int)

            #indices_pre = self._random.sample(self._index_range, batch_len)
            #indices_pre = np.array(indices_pre)
            #indices = (indices_pre + self._curr_insert_ptr) % self._curr_len

        else:
            cdf = np.cumsum(self._hist_error+0.01)
            cdf = cdf / cdf[-1]
            values = np.random.rand(batch_len)
            indices = np.searchsorted(cdf, values)




        # states = self._hist_St[indices]
        # actions = self._hist_At[indices]
        # rewards_1 = self._hist_Rt_1[indices]
        # states_1 = self._hist_St_1[indices]
        # dones = self._hist_done[indices]

        # pdb.set_trace()

        states = np.take(self._hist_St, indices, axis=0)
        actions = np.take(self._hist_At, indices, axis=0)
        rewards_1 = np.take(self._hist_Rt_1, indices, axis=0)
        states_1 = np.take(self._hist_St_1, indices, axis=0)
        dones = np.take(self._hist_done, indices, axis=0)

        # batch = []
        # for i, idx in enumerate(indices):
        #     tup = (self._hist_St[idx],
        #            self._hist_At[idx],
        #            self._hist_Rt_1[idx],
        #            self._hist_St_1[idx],
        #            self._hist_done[idx])
        #     batch.append(tup)

        #     assert (tup[0] == states[i]).all()
        #     assert (tup[1] == actions[i]).all()
        #     assert (tup[2] == rewards_1[i]).all()
        #     assert (tup[3] == states_1[i]).all()
        #     assert (tup[4] == dones[i]).all()

        # pdb.set_trace()

        return states, actions, rewards_1, states_1, dones, indices

    def update_errors(self, indices, errors):
        assert isinstance(indices, np.ndarray)
        assert indices.ndim == 1
        assert len(indices) > 0
        assert isinstance(errors, np.ndarray)
        assert errors.ndim == 1
        assert len(indices) == len(errors)

        self._hist_error[indices] = np.abs(errors)

if __name__ == '__main__':

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
