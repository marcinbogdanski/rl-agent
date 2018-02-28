import numpy as np
import tensorflow as tf

import pdb

# TODO: Documentation
# TODO: Proper unittest
# TODO: update normalisation, requires updating unittest

class KerasApproximator:

    def __init__(self, discount, model):
        self._discount = discount
        self._model = model
        self._state_space = None
        self._action_space = None

    def set_state_action_spaces(self, state_space, action_space):
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
        assert self._state_space is not None
        assert self._action_space is not None
        assert isinstance(states, np.ndarray)
        assert states.shape[1:] == self._state_space.shape
        # assert [self._state_space.contains(x) for x in states]
        
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
        # assert [self._state_space.contains(x) for x in states]

        states = np.copy(states)
        states -= self._offsets
        states *= self._scales

        # TODO: use this form and update unittest
        # states_norm = (states - self._offsets) * self._scales

        return self._model.predict(states, batch_size=len(states))

        
    def update2(self, states, actions, rewards_n, states_n, dones):
        assert self._state_space is not None
        assert self._action_space is not None

        assert isinstance(states, np.ndarray)
        assert states.shape[1:] == self._state_space.shape
        # assert all(map(self._state_space.contains, states))

        assert isinstance(actions, np.ndarray)
        assert actions.shape[1:] == self._action_space.shape
        # assert all(map(self._action_space.contains, actions))

        assert isinstance(rewards_n, np.ndarray)
        assert rewards_n.dtype == float
        assert rewards_n.ndim == 1

        assert isinstance(states_n, np.ndarray)
        assert states_n.shape[1:] == self._state_space.shape
        # assert all(map(self._state_space.contains, states_n))

        assert isinstance(dones, np.ndarray)
        assert dones.dtype == bool
        assert dones.ndim == 1

        assert len(states) == len(actions) \
            == len(rewards_n) == len(states_n) == len(dones)


        #
        #   This chunk could use some comments
        #
        
        inputs = np.copy(states)
        inputs_n = np.copy(states_n)
        not_dones = np.logical_not(dones)

        inputs -= self._offsets
        inputs *= self._scales
        inputs_n -= self._offsets
        inputs_n *= self._scales

        # TODO: use this form and update unittest
        # inputs = (states - self._offsets) * self._scales
        # inputs_n = (states_n - self._offsets) * self._scales
        
        # Join arrays for single predict() call (double speed improvement)
        inputs_joint = np.concatenate((inputs, inputs_n))
        result_joint = self._model.predict(inputs_joint, batch_size=len(inputs_joint))
        targets, est_n = np.split(result_joint, 2)
        
        q_n = np.max(est_n, axis=1, keepdims=True).flatten()
        tt = rewards_n + (not_dones * self._discount * q_n)
        errors = tt - targets[np.arange(len(targets)), actions]
        targets[np.arange(len(targets)), actions] = tt
        
        #self._model.train_on_batch(inputs, targets)
        self._model.fit(inputs, targets, batch_size=len(inputs), epochs=1, verbose=False)
        
        return errors