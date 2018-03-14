import numpy as np
import gym

class PolicyRandom:
    def __init__(self):
        self._state_space = None
        self._action_space = None

    def set_state_action_spaces(self, state_space, action_space):
        self._state_space = state_space
        self._action_space = action_space

    def link(self, agent):
        pass

    def reset(self):
        pass

    def next_step(self, total_step):
        pass

    def pick_action(self, state):
        return self._action_space.sample()

    def train_single(self, state, action, target):
        pass

    def train_batch(self, states, actions, targets):
        pass

    def get_raw(self):
        pass