import numpy as np
import pdb

class ApproximatorBase:
    def __init__(self):
        """Base class for all aproximators

        Currently tested on continous 2d state spaces with categorical actions
        """
        self._logger = None
        self._log_every = None
        self._log_samples = None

    def set_state_action_spaces(self, state_space, action_space):
        """Set state/action space descriptors
        Params:
            state_space: gym.spaces.Box (tested) or Discrete (not tested)
            action_space: gym.spaces.Box (not tested) or Discrete (tested)
        """
        raise NotImplemented()

    def get_weights_fingerprint(self):
        """Unique numer representing this object, e.g. sum of weights"""
        raise NotImplemented()

    def estimate(self, state, action):
        """Estimate single state/action pair

        Args:
            state - state as provided by environment
            action - as required by environment

        Returns:
            float: Q-Value for that state/action pair
        """
        raise NotImplemented()

    def train(self, states, actions, targets):
        """Train approximator to be closer to targets

        Args:
            states: array of states, states[0..n] must be a valid state
            actions: array of actions, actions[0..n] must be a valid action
            targets: array of floats

        Returns:
            None
        """
        raise NotImplemented()

    def install_logger(self, logger, log_every, samples):
        self._logger = logger
        self._log_every = log_every
        self._log_samples = samples

    def perform_logging(self, episode, step, total_step):
        assert self._state_space is not None
        assert self._action_space is not None

        if self._logger is not None and not self._logger.is_initialized:
            self._logger.add_data_item('q_val')

        #
        #   Log Q-Values
        #
        if self._logger is not None and total_step % self._log_every == 0:

            if self._state_space.shape != (2,):
                raise ValueError('Can only log 2d state space')

            low = self._state_space.low
            high = self._state_space.high
            nb_sampl = self._log_samples
            linspace_dim_1 = np.linspace(low[0], high[0], nb_sampl[0])
            linspace_dim_2 = np.linspace(low[1], high[1], nb_sampl[1])

            num_tests = len(linspace_dim_1) * len(linspace_dim_2)
            d_1_skip = len(linspace_dim_2)
            states = np.zeros([num_tests, 2])
            for d_1 in range(len(linspace_dim_1)):
                for d_2 in range(len(linspace_dim_2)):
                    states[d_1*d_1_skip + d_2, 0] = linspace_dim_1[d_1]
                    states[d_1*d_1_skip + d_2, 1] = linspace_dim_2[d_2]


            q_list = self.estimate_all(states)
            q_val = np.zeros([len(linspace_dim_1), len(linspace_dim_2), self._action_space.n])
            
            for si in range(len(states)):    
                d_1 = si//d_1_skip
                d_2 = si %d_1_skip
                q_val[d_1, d_2] = q_list[si]

            self._logger.append(episode, step, total_step, q_val=q_val)