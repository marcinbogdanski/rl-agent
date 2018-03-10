
import numpy as np

import gym

import pdb

# TODO: Under Construction

class Gridworld:
    """Simple gridworld

        States are numbered as follows:
              [ 12, 13, 14, 15]
              [  8,  9, 10, 11]
            Y [  4,  5,  6,  7]
              [  0,  1,  2,  3]
                       X
        where state x=0, y=0 is in bottom left
    """

    def __init__(self, size_x, size_y,
            random_start=False,
            default_reward=-1):
        self._size_x = size_x
        self._size_y = size_y
        self._random_start = random_start

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Discrete(size_x*size_y)

        states = np.arange(size_x*size_y, dtype=int)
        self._states = states.reshape(size_y, size_x)
        self._grid = np.zeros([size_y, size_x], dtype=int)
        self._act_rewards = \
            np.zeros([size_y, size_x, 4], dtype=float) + default_reward

        self._start_states = []

        self._agent_x = None
        self._agent_y = None

        self._is_done = False


    def set_state(self, x, y, type_):
        if type_ == 'start':
            self._grid[y, x] = 1
            self._start_states.append(self._states[y, x])
        if type_ == 'terminal':
            self._grid[y, x] = 2


    def reset(self):
        self._is_done = False

        if self._random_start:
            while True:
                state = np.random.randint(0, self._size_x*self._size_y)
                x, y = state % self._size_y, state // self._size_x
                if self._grid[y, x] == 0:  # neutral state
                    self._agent_y = y
                    self._agent_x = x
                    break
        else:
            state = np.random.choice(self._start_states)
            self._agent_y = state // self._size_x
            self._agent_x = state % self._size_y
        return self._states[self._agent_y, self._agent_x]


    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError('Invalid action')
        if self._is_done:
            raise ValueError('Episode terminated')

        reward = self._act_rewards[self._agent_y, self._agent_x, action]

        if action == 0 and self._agent_y < self._size_x-1:  # NORTH
            self._agent_y += 1
        elif action == 1 and self._agent_x < self._size_x-1:  # EAST
            self._agent_x += 1
        elif action == 2 and self._agent_y > 0:  # SOUTH
            self._agent_y -= 1
        elif action == 3 and self._agent_x > 0:  # WEST
            self._agent_x -= 1

        if self._grid[self._agent_y, self._agent_x] == 2:  # terminal
            self._is_done = True

        return self._states[self._agent_y, self._agent_x], reward, self._is_done


    def _print_state(self, y, x):
        is_agent = False
        if self._agent_x == x and self._agent_y == y:
            print('A', end='')
        elif self._grid[y, x] == 2:  # terminal
            print('T', end='')
        else:
            print('.', end='')

    def _print_row(self, y):
        for x in range(self._size_x):
            self._print_state(y, x)
        print()

    def render(self):

        # print('--- env ---')
        # print('Agent pos: [', self._agent_x, self._agent_y, ']')
        for y in range(self._size_y-1, -1, -1):
            self._print_row(y)



