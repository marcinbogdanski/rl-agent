import numpy as np
import tensorflow as tf
from .approx_base import ApproximatorBase
import gym

import pdb

# TODO: If state/action space is discretee, auto-one-hot-encode?
# TODO: Add ability to configure normalization?
# TODO: Documentation
# TODO: Proper unittest
# TODO: update normalisation code, requires updating unittest

class QFunctKeras(ApproximatorBase):

    def __init__(self, model):
        """Q-function approximator using Keras model

        Tested for continous 2d observation space and categorical actions

        See base class documentation for method descriptions

        Args:
            model: Keras compiled model
        """
        super().__init__()
        self._model = model
        self._state_space = None
        self._action_space = None

    def set_state_action_spaces(self, state_space, action_space):

        # These should be relaxed in the future,
        # possibly remove gym dependancy
        if not isinstance(state_space, gym.spaces.Box):
            raise ValueError('Only gym.spaces.Box state space supproted')
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError('Only gym.spaces.Discrete action space supported')

        first_layer = self._model.layers[0]
        nn_input_shape = first_layer.input_shape[1:]
        if state_space.shape != nn_input_shape:
            raise ValueError('Input shape does not match state_space shape')

        last_layer = self._model.layers[-1]
        nn_output_shape = action_space.shape
        if action_space.shape != nn_output_shape:
            raise ValueError('Output shape does not match action_space shape')

        self._state_space = state_space
        self._action_space = action_space

        # normalise inputs
        low = state_space.low
        high = state_space.high
        self._offsets = low + (high - low) / 2
        self._scales = 1 / ((high - low) / 2)
        

    def get_weights_fingerprint(self):
        weights_sum = 0
        for idx, layer in enumerate(self._model.layers):
            list_weights = layer.get_weights()
            layer_sum = np.sum(np.sum(ii) for ii in list_weights)
            weights_sum += layer_sum
        return weights_sum


    def estimate(self, state, action):
        assert False
        assert self._state_space is not None
        assert self._action_space is not None
        assert isinstance(states, np.ndarray)
        assert states.shape[1:] == self._state_space.shape
        # assert all(map(self._state_space.contains, states))
        
        states = np.copy(states)
        states -= self._offsets
        states *= self._scales

        est = self._model.predict(states)


        return est[0, action]

    def estimate_all(self, states):
        assert self._state_space is not None
        assert self._action_space is not None
        assert isinstance(states, np.ndarray)
        assert states.shape[1:] == self._state_space.shape
        # assert all(map(self._state_space.contains, states))

        states = np.copy(states)
        states -= self._offsets
        states *= self._scales

        # TODO: use this form and update unittest
        # states_norm = (states - self._offsets) * self._scales

        return self._model.predict(states, batch_size=len(states))

        

    def max_op(self, states):
        assert self._state_space is not None
        assert self._action_space is not None

        assert isinstance(states, np.ndarray)
        assert states.shape[1:] == self._state_space.shape
        # assert all(map(self._state_space.contains, states))

        # TODO: use this form and update unittest
        # inputs = (states - self._offsets) * self._scales
        inputs = np.copy(states)
        inputs -= self._offsets
        inputs *= self._scales
        outputs = self._model.predict(inputs, batch_size=len(inputs))

        out_max = np.max(outputs, axis=1)

        assert out_max.ndim == 1
        assert len(out_max) == len(states)

        return out_max

    def train(self, states, actions, targets):
        assert self._state_space is not None
        assert self._action_space is not None

        assert isinstance(states, np.ndarray)
        assert states.shape[1:] == self._state_space.shape
        # assert all(map(self._state_space.contains, states))

        assert isinstance(actions, np.ndarray)
        assert actions.shape[1:] == self._action_space.shape
        # assert all(map(self._action_space.contains, actions))

        assert isinstance(targets, np.ndarray)
        assert targets.ndim == 1

        assert len(states) == len(actions) == len(targets)




        # TODO: use this form and update unittest
        # inputs = (states - self._offsets) * self._scales
        inputs = np.copy(states)
        inputs -= self._offsets
        inputs *= self._scales
        all_targets = self._model.predict(inputs, batch_size=len(inputs))
        
        

        errors = targets - all_targets[np.arange(len(all_targets)), actions]
        all_targets[np.arange(len(all_targets)), actions] = targets
        
        #self._model.train_on_batch(inputs, all_targets)
        self._model.fit(inputs, all_targets, batch_size=len(inputs), epochs=1, verbose=False)

        return errors