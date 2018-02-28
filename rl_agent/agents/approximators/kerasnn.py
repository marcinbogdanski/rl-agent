import numpy as np
import tensorflow as tf

import pdb

class KerasApproximator:

    def __init__(self, discount, model):
        self._discount = discount
        self._model = model
        self._state_space = None
        self._action_space = None

        self._pos_offset = 0.3 # 0.35
        self._pos_scale = 1 / 0.9  # 2 / 1.7  # -1.2 to 0.5 should be for NN
        self._vel_scale = 1 / 0.07 # 2 / 0.14  # maps vel to -1..1

    def set_state_action_spaces(self, state_space, action_space):
        self._state_space = state_space
        self._action_space = action_space

    def get_weights_fingerprint(self):
        weights_sum = 0
        for idx, layer in enumerate(self._model.layers):
            list_weights = layer.get_weights()
            layer_sum = np.sum(np.sum(ii) for ii in list_weights)
            weights_sum += layer_sum
        return weights_sum

    def _test_input(self, state, action):
        assert isinstance(state, np.ndarray)
        assert isinstance(state[0], float)
        assert isinstance(state[1], float)
        assert isinstance(action, int) or isinstance(action, np.int64)
        pos, vel = state[0], state[1]
        assert -1.2 <= pos and pos <= 0.5
        assert -0.07 <= vel and vel <= 0.07

        return pos, vel, action

    def estimate(self, state, action):
        assert self._state_space is not None
        assert self._action_space is not None
        pos, vel, action = self._test_input(state, action)

        pos += self._pos_offset
        pos *= self._pos_scale
        vel *= self._vel_scale

        est = self._model.predict(np.array([[pos, vel]]))


        return est[0, action]

    def estimate_all(self, states):
        assert self._state_space is not None
        assert self._action_space is not None
        assert isinstance(states, np.ndarray)
        assert states.ndim == 2
        assert len(states) > 0
        assert states.shape[1] == 2   # pos, vel
        assert states.dtype == np.float32 or states.dtype == np.float64
        assert np.min(states, axis=0)[0] >= -1.2  # pos
        assert np.max(states, axis=0)[0] <= 0.5  # pos
        assert np.min(states, axis=0)[1] >= -0.07  # vel
        assert np.max(states, axis=0)[1] <= 0.07  # vel

        states = np.array(states)

        states[:,0] += self._pos_offset
        states[:,0] *= self._pos_scale
        states[:,1] *= self._vel_scale

        est = self._model.predict(states, batch_size=len(states))
        return est  # return 2d array

    def update(self, batch):
        assert self._state_space is not None
        assert self._action_space is not None
        assert isinstance(batch, list)
        assert len(batch) > 0
        assert len(batch[0]) == 5

        inputs = []
        targets = []
        for tup in batch:
            St = tup[0]
            At = tup[1]
            Rt_1 = tup[2]
            St_1 = tup[3]
            done = tup[4]

            pp = St[0]
            vv = St[1]
            aa = At

            rr_n = Rt_1
            pp_n = St_1[0]
            vv_n = St_1[1]

            pp += self._pos_offset
            pp *= self._pos_scale
            vv *= self._vel_scale

            pp_n += self._pos_offset
            pp_n *= self._pos_scale
            vv_n *= self._vel_scale

            inp = np.array([[pp, vv]])
            inp_n = np.array([[pp_n, vv_n]])

            est = self._model.predict(inp, batch_size=len(inp))
            est_n = self._model.predict(inp_n, batch_size=len(inp_n))
            q_n = np.max(est_n)

            if done:
                tt = rr_n 
            else:
                tt = rr_n + self._discount * q_n

            assert aa in [0, 1, 2]
            assert est.shape[0] == 1
            est[0, aa] = tt

            inputs.append([pp, vv])
            targets.append(est[0])
        
        inputs = np.array(inputs)
        targets = np.array(targets)
        
        self._model.train_on_batch(inputs, targets)
        
    def update2(self, states, actions, rewards_n, states_n, dones):
        assert self._state_space is not None
        assert self._action_space is not None
        assert isinstance(states, np.ndarray)
        assert states.dtype == float
        assert states.ndim == 2
        assert states.shape[0] >= 1

        assert isinstance(actions, np.ndarray)
        assert actions.dtype == int
        assert (actions.ndim == 2 and actions.shape[0] >= 1 and actions.shape[1] == 1) \
                or (actions.ndim == 1 and actions.shape[0] >= 1)

        assert isinstance(rewards_n, np.ndarray)
        assert rewards_n.dtype == float
        assert rewards_n.ndim == 1
        assert rewards_n.shape[0] >= 1

        assert isinstance(states_n, np.ndarray)
        assert states_n.dtype == float
        assert states_n.ndim == 2
        assert states_n.shape[0] >= 1

        assert isinstance(dones, np.ndarray)
        assert dones.dtype == bool
        assert dones.ndim == 1
        assert dones.shape[0] >= 1



        # assert len(batch) > 0
        # assert len(batch[0]) == 5

        # inputs = np.zeros([len(batch), 2], dtype=np.float32)
        # actions = np.zeros([len(batch)], dtype=np.int8)
        # rewards_n = np.zeros([len(batch), 1], dtype=np.float32)
        # inputs_n = np.zeros([len(batch), 2], dtype=np.float32)
        # not_dones = np.zeros([len(batch), 1], dtype=np.bool)
        
        inputs = np.array(states)
        inputs_n = np.array(states_n)
        not_dones = np.logical_not(dones)
        
        # for i, tup in enumerate(batch):
        #     St = tup[0]
        #     At = tup[1]
        #     Rt_1 = tup[2]
        #     St_1 = tup[3]
        #     done = tup[4]

        #     inputs[i] = St
        #     actions[i] = At
        #     rewards_n[i] = Rt_1
        #     inputs_n[i] = St_1
        #     not_dones[i] = not done

        #     assert At in [0, 1, 2]
        
        inputs[:,0] += self._pos_offset
        inputs[:,0] *= self._pos_scale
        inputs[:,1] *= self._vel_scale
        inputs_n[:,0] += self._pos_offset
        inputs_n[:,0] *= self._pos_scale
        inputs_n[:,1] *= self._vel_scale
        
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