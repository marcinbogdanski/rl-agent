import numpy as np

from . import neural_mini

import pdb

class NeuralApproximator:

    def __init__(self, step_size, discount, batch_size, log=None):
        self._step_size = step_size
        self._discount = discount
        self._batch_size = batch_size

        self._nn = neural_mini.NeuralNetwork2([2, 128, 3])

        self._pos_offset = 0.35
        self._pos_scale = 2 / 1.7  # -1.2 to 0.5 should be for NN
        self._vel_scale = 2 / 0.14  # maps vel to -1..1

        if log is not None:
            log.add_param('type', 'neural network')
            log.add_param('nb_inputs', 2)
            log.add_param('hid_1_size', 128)
            log.add_param('hid_1_act', 'sigmoid')
            log.add_param('out_size', 3)
            log.add_param('out_act', 'linear')

    def _test_input(self, state, action):
        assert isinstance(state, np.ndarray)
        assert isinstance(state[0], float)
        assert isinstance(state[1], float)
        assert isinstance(action, int) or isinstance(action, np.int64)
        pos, vel = state[0], state[1]
        assert -1.2 <= pos and pos <= 0.5
        assert -0.07 <= vel and vel <= 0.07
        assert action in [0, 1, 2]

        return pos, vel, action

    def estimate(self, state, action):
        pos, vel, action = self._test_input(state, action)

        pos += self._pos_offset
        pos *= self._pos_scale
        vel *= self._vel_scale

        est = self._nn.forward(np.array([[pos, vel]]))

        assert action in [0, 1, 2]

        return est[0, action]

    def estimate_all(self, states):
        assert isinstance(states, np.ndarray)
        assert states.ndim == 2
        assert len(states) > 0
        assert states.shape[1] == 2   # pos, vel
        assert states.dtype == np.float32 or states.dtype == np.float64
        assert np.min(states, axis=0)[0] >= -1.2  # pos
        assert np.max(states, axis=0)[0] <= 0.5  # pos
        assert np.min(states, axis=0)[1] >= -0.07  # vel
        assert np.max(states, axis=0)[1] <= 0.07  # vel

        states[:,0] += self._pos_offset
        states[:,0] *= self._pos_scale
        states[:,1] *= self._vel_scale

        est = self._nn.forward(states)
        return est  # return 2d array

    def update(self, batch, timing_dict):
        assert isinstance(batch, list)
        assert len(batch) > 0
        assert len(batch[0]) == 5

        time_start = time.time()
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

            time_pred = time.time()
            est = self._nn.forward(inp)
            est_n = self._nn.forward(inp_n)
            timing_dict['      update_loop_pred'] += time.time() - time_pred

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
        timing_dict['    update_loop'] += time.time() - time_start

        time_start = time.time()
        inputs = np.array(inputs)
        targets = np.array(targets)
        timing_dict['    update_convert_numpy'] += time.time() - time_start

        time_start = time.time()
        self._nn.train_batch(inputs, targets, self._step_size)
        timing_dict['    update_train_on_batch'] += time.time() - time_start

    def update2(self, batch, timing_dict):
        assert isinstance(batch, list)
        assert len(batch) > 0
        assert len(batch[0]) == 5

        time_start = time.time()
        inputs = np.zeros([len(batch), 2], dtype=np.float32)
        actions = np.zeros([len(batch)], dtype=np.int8)
        rewards_n = np.zeros([len(batch), 1], dtype=np.float32)
        inputs_n = np.zeros([len(batch), 2], dtype=np.float32)
        not_dones = np.zeros([len(batch), 1], dtype=np.bool)
        timing_dict['    update2_create_arr'] += time.time() - time_start

        time_start = time.time()
        for i, tup in enumerate(batch):
            St = tup[0]
            At = tup[1]
            Rt_1 = tup[2]
            St_1 = tup[3]
            done = tup[4]

            inputs[i] = St
            actions[i] = At
            rewards_n[i] = Rt_1
            inputs_n[i] = St_1
            not_dones[i] = not done

            assert At in [0, 1, 2]
        timing_dict['    update2_loop'] += time.time() - time_start

        time_start = time.time()
        inputs[:,0] += self._pos_offset
        inputs[:,0] *= self._pos_scale
        inputs[:,1] *= self._vel_scale
        inputs_n[:,0] += self._pos_offset
        inputs_n[:,0] *= self._pos_scale
        inputs_n[:,1] *= self._vel_scale
        timing_dict['    update2_scale'] += time.time() - time_start

        time_start = time.time()
        targets = self._nn.forward(inputs)
        est_n = self._nn.forward(inputs_n)
        timing_dict['    update2_predict'] += time.time() - time_start

        time_start = time.time()
        q_n = np.max(est_n, axis=1, keepdims=True)
        tt = rewards_n + (not_dones * self._discount * q_n)
        targets[np.arange(len(targets)), actions] = tt.flatten()
        timing_dict['    update2_post'] += time.time() - time_start

        time_start = time.time()
        self._nn.train_batch(inputs, targets, self._step_size)
        timing_dict['    update2_train_on_batch'] += time.time() - time_start

