
import numpy as np

import gym

import pdb

# TODO: Under Construction

class Corridor:
    """Simple corridor world

        States are numbered as follows:
              [0, 1, 2, 3, ...]

        Left-most and right-most states are terminal
    """

    def __init__(self, length, start_state, default_reward=0):
        self._length = length
        self._start_state = start_state

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(length)

        self._states = np.zeros([length], dtype=int)
        self._states[0] = 2   # terminal
        self._states[-1] = 2  # terminal
        self._rewards = \
            np.zeros([length, 2], dtype=float) + default_reward

        self._agent_pos = None
        self._is_done = False


    def reset(self):
        self._is_done = False
        self._agent_pos = self._start_state
        return self._agent_pos


    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError('Invalid action')
        if self._is_done:
            raise ValueError('Episode is terminated')

        reward = self._rewards[self._agent_pos, action]

        if action == 0 and self._agent_pos > 0:    # LEFT
            self._agent_pos -= 1
        elif action == 1 and self._agent_pos < self._length-1:  # RIGHT
            self._agent_pos += 1

        if self._states[self._agent_pos] == 2:  # terminal
            self._is_done = True

        return self._agent_pos, reward, self._is_done


    def _print_state(self, pos):
        if self._agent_pos == pos:
            print('A', end='')
        elif self._states[pos] == 2:  # terminal
            print('T', end='')
        else:
            print('.', end='')

    def render(self):
        for p in range(self._length):
            self._print_state(p)
        print()



