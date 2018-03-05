import numpy as np

class VanillaPolicyGrad:
    def __init__(self):
        """Under construction"""
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
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError('Only gym.spaces.Discrete action space supported')

        self._state_space = state_space
        self._action_space = action_space

    def link(self, agent):
        if hasattr(agent, 'V'):
            self._v_approx = agent.V
        if hasattr(agent, 'Q'):
            self._q_approx = agent.Q


    def reset(self):
        """Reset at the end of episode"""
        pass

    def next_step(self, total_step):
        pass

    def pick_action(self, state):
        raise NotImplemented()

    def train(self):
        raise NotImplemented