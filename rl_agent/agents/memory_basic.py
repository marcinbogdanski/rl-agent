import numpy as np
import gym
import pdb

class MemoryBasic:
    """Buffer for actor-critic agents"""

    def __init__(self, max_len):
        """
        Args:
            max_len: maximum capacity
        """
        assert isinstance(max_len, int)
        assert max_len > 0

        self._max_len = max_len
        self._curr_insert_ptr = 0

        self._nb_episodes = 0


    @property
    def nb_episodes(self):
        return self._nb_episodes


    def set_state_action_spaces(self, state_space, action_space):
        """Set state/action space descriptors
        Params:
            state_space: gym.spaces.Box (tested) or Discrete (not tested)
            action_space: gym.spaces.Box (not tested) or Discrete (tested)
        """
            
        # These should be relaxed in the future to support more spaces,
        # possibly remove gym dependancy
        if not isinstance(state_space, (gym.spaces.Discrete, gym.spaces.Box)):
            raise ValueError('Only Discrete and Box state spaces supproted')
        if not isinstance(action_space, (gym.spaces.Discrete, gym.spaces.Box)):
            raise ValueError('Only Discrete and Box action spaces supported')

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

        self._hist_St = np.zeros(St_shape, dtype=state_space.dtype)
        self._hist_At = np.zeros(At_shape, dtype=action_space.dtype)
        self._hist_Rt_1 = np.zeros(Rt_1_shape, dtype=float)
        self._hist_St_1 = np.zeros(St_1_shape, dtype=state_space.dtype)
        self._hist_done = np.zeros(done_shape, dtype=bool)


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

        self._curr_insert_ptr += 1

        if self._curr_insert_ptr >= self._max_len:
            raise ValueError('Memory capacity exceeded')

        if done:
            self._nb_episodes += 1

    def clear(self):
        self._curr_insert_ptr = 0
        self._nb_episodes = 0
        self._hist_St.fill(0)
        self._hist_At.fill(0)
        self._hist_Rt_1.fill(0)
        self._hist_St_1.fill(0)
        self._hist_done.fill(0)

    def length(self):
        return self._curr_insert_ptr

    def get_data(self):
        return \
            self._hist_St[0:self._curr_insert_ptr], \
            self._hist_At[0:self._curr_insert_ptr], \
            self._hist_Rt_1[0:self._curr_insert_ptr], \
            self._hist_St_1[0:self._curr_insert_ptr], \
            self._hist_done[0:self._curr_insert_ptr]


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

    pdb.set_trace()

    mem = EPBuffer(max_len=3)

    mem.set_state_action_spaces(
        state_space=gym.spaces.Box(low=0, high=5, shape=(2,)),
        action_space=gym.spaces.Discrete(2))

    mem.append(
        St=np.array([1, 1]),
        At=1,
        Rt_1=1,
        St_1=np.array([2, 2]),
        done=False)