import gym

class SwapActions:
    """Test envirnoment corresponds to Sutton and Barto 13.1 Nov 2017 draft.

    This is 3-state environment with middle state having left-right actions
    swapped.

    All states look the same to the agent.

    States are as follows:
     - 0 - left most state, start state, action left has no effect
     - 1 - middle state, has swapped left-right actions
     - 2 - right state, action right terminates episode

    Best policy is to move right with prob approx 0.59
    Best cumulative reward is approx -12
    """

    def __init__(self):
        self._state = 0
        self._is_done = False

        # 0 is move left, 1 is move right, swapped in central state
        self.action_space = gym.spaces.Discrete(2)
        
        # all states look the same, so we always return obs equal zero
        self.observation_space = gym.spaces.Discrete(1) 

    def reset(self):
        self._state = 0
        return 0  # observation is always zero

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError('Invalid action')
        if self._is_done:
            raise ValueError('Episode is terminated')
        
        if self._state == 0:
            # Left-most sate, NORMAL actions
            if action == 0:
                # do nothing
                return 0, -1, False
            else:
                self._state += 1
                return 0, -1, False
        elif self._state == 1:
            # Mid state, SWAPED actions
            if action == 0:
                self._state += 1  # go right
                return 0, -1, False
            else:
                self._state -= 1  # go left
                return 0, -1, False
        elif self._state == 2:
            # Righ-most state, NORMAL actions
            if action == 0:
                self._state -= 1  # go left
                return 0, -1, False
            else:
                self._state += 1  # go right
                return 0, -1, True
        else:
            raise ValueError('Invalid state')

    def render(self):
        print('current hidden state:', self._state)        