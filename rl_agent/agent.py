import numpy as np
import matplotlib.pyplot as plt
import collections
import time
import pdb
import random
import math

from . import tile_coding
from . import neural_mini
from . import memory

# from keras import Sequential
# from keras.layers import Dense
# from keras.optimizers import RMSprop, sgd
import tensorflow as tf



def _rand_argmax(vector):
    """ Argmax that chooses randomly among eligible maximum indices. """
    vmax = np.max(vector)
    indices = np.nonzero(vector == vmax)[0]
    return np.random.choice(indices)


class AggregateApproximator:
    def __init__(self, step_size, action_space, init_val=0, log=None):
        self._step_size = step_size
        self._action_space = action_space
        
        eps = 1e-5

        self._pos_bin_nb = 64
        self._pos_bins = np.linspace(-1.2, 0.5+eps, self._pos_bin_nb+1)

        self._vel_bin_nb = 64
        self._vel_bins = np.linspace(-0.07, 0.07+eps, self._vel_bin_nb+1)

        self._action_nb = 3

        self._states = np.zeros([self._pos_bin_nb,
            self._vel_bin_nb, self._action_nb]) + init_val

    def get_weights_fingerprint(self):
        return np.sum(self._states)

    def _to_idx(self, state, action):
        assert isinstance(state, np.ndarray)
        assert isinstance(state[0], float)
        assert isinstance(state[1], float)
        assert isinstance(action, int) or isinstance(action, np.int64)

        pos, vel = state[0], state[1]

        assert -1.2 <= pos and pos <= 0.5
        assert -0.07 <= vel and vel <= 0.07

        assert action in [0, 1, 2]
        act_idx = action

        pos_idx = np.digitize(pos, self._pos_bins) - 1
        if vel == 0.07:
            vel_idx = self._vel_bin_nb-1
        else:
            vel_idx = np.digitize(vel, self._vel_bins) - 1

        assert 0 <= pos_idx and pos_idx <= self._pos_bin_nb-1
        assert 0 <= vel_idx and vel_idx <= self._vel_bin_nb-1

        return pos_idx, vel_idx, act_idx

    def estimate(self, state, action):
        pos_idx, vel_idx, act_idx = self._to_idx(state, action)
        return self._states[pos_idx, vel_idx, act_idx]
    
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

        result = np.zeros( [len(states), len(self._action_space)], dtype=float)
        for si in range(len(states)):
            for i in range(len(self._action_space)):
                action = self._action_space[i]
                result[si, i] = self.estimate(states[si], action)

        return result

    def update(self, state, action, target):
        pos_idx, vel_idx, act_idx = self._to_idx(state, action)

        pos = state[0]
        assert pos < 0.5  # this should never be called on terminal state
        
        est = self.estimate(state, action)
        
        self._states[pos_idx, vel_idx, act_idx] += \
            self._step_size * (target - est)

    def update2(self, states, actions, rewards_n, states_n, dones, timing_dict):

        # pdb.set_trace()
        # print('hop')

        est_arr = self.estimate_all(states)
        est_arr_1 = self.estimate_all(states_n)

        errors = np.zeros([len(states)])

        for i in range(len(states)):
            St = states[i]
            At = actions[i, 0]
            Rt_1 = rewards_n[i, 0]
            St_1 = states_n[i]
            done = dones[i, 0]

            est = est_arr_1[i]
            At_1 = _rand_argmax(est)

            if done:
                Tt = Rt_1
            else:
                Tt = Rt_1 + 0.99 * self.estimate(St_1, At_1)

            errors[i] = Tt - est_arr[i, At]

            self.update(St, At, Tt)

        return errors
        


class TileApproximator:

    def __init__(self, step_size, action_space, init_val=0, log=None):
        self._num_of_tillings = 8
        self._step_size = step_size / self._num_of_tillings

        self._action_space = action_space

        self._pos_scale = self._num_of_tillings / (0.5 + 1.2)
        self._vel_scale = self._num_of_tillings / (0.07 + 0.07)

        self._hashtable = tile_coding.IHT(2048)
        self._weights = np.zeros(2048) + init_val / self._num_of_tillings
        
        max_len = 2000

    def get_weights_fingerprint(self):
        return np.sum(self._weights)

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

        active_tiles = tile_coding.tiles(
            self._hashtable, self._num_of_tillings,
            [self._pos_scale * pos, self._vel_scale * vel], [action])

        return np.sum(self._weights[active_tiles])

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

        result = np.zeros( [len(states), len(self._action_space)], dtype=float)
        for si in range(len(states)):
            for i in range(len(self._action_space)):
                action = self._action_space[i]
                result[si, i] = self.estimate(states[si], action)

        return result


    def update(self, state, action, target):
        pos, vel, action = self._test_input(state, action)
        assert pos < 0.5  # this should never be called on terminal state

        active_tiles = tile_coding.tiles(
            self._hashtable, self._num_of_tillings,
            [self._pos_scale * pos, self._vel_scale * vel],
            [action])

        est = np.sum(self._weights[active_tiles])

        delta = self._step_size * (target - est)

        for tile in active_tiles:
            self._weights[tile] += delta

    def update2(self, states, actions, rewards_n, states_n, dones, timing_dict):

        # pdb.set_trace()
        # print('hop')

        est_arr = self.estimate_all(states)
        est_arr_1 = self.estimate_all(states_n)

        errors = np.zeros([len(states)])

        for i in range(len(states)):
            St = states[i]
            At = actions[i, 0]
            Rt_1 = rewards_n[i, 0]
            St_1 = states_n[i]
            done = dones[i, 0]

            est = est_arr_1[i]
            At_1 = _rand_argmax(est)

            if done:
                Tt = Rt_1
            else:
                Tt = Rt_1 + 0.99 * self.estimate(St_1, At_1)

            errors[i] = Tt - est_arr[i, At]

            self.update(St, At, Tt)

        return errors


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


class KerasApproximator:

    def __init__(self, input_count, output_count, step_size, 
            discount, batch_size, log=None):
        self._input_count = input_count
        self._output_count = output_count
        self._step_size = step_size
        self._discount = discount
        self._batch_size = batch_size

        # self._model = Sequential()
        # # self._model.add(Dense(output_dim=128, activation='sigmoid', input_dim=2))
        # # self._model.add(Dense(output_dim=3, activation='linear'))
        # self._model.add(Dense(activation='sigmoid', input_dim=2, units=128))
        # self._model.add(Dense(activation='linear', units=3))
        # # self._model.compile(loss='mse', optimizer=RMSprop(lr=0.00025))
        # self._model.compile(loss='mse', optimizer=sgd(lr=0.01))


        # self._model = tf.keras.models.Sequential()
        # self._model.add(tf.keras.layers.Dense(activation='sigmoid', input_dim=2, units=128))
        # self._model.add(tf.keras.layers.Dense(activation='linear', units=3))
        # self._model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(lr=0.01))

        self._model = tf.keras.models.Sequential()
        self._model.add(tf.keras.layers.Dense(units=256, activation='relu', input_dim=input_count))
        self._model.add(tf.keras.layers.Dense(units=256, activation='relu'))
        self._model.add(tf.keras.layers.Dense(units=output_count, activation='linear'))
        # self._model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(lr=0.001))
        opt = tf.keras.optimizers.RMSprop(lr=step_size)
        self._model.compile(loss='mse', optimizer=opt)

        self._pos_offset = 0.3 # 0.35
        self._pos_scale = 1 / 0.9  # 2 / 1.7  # -1.2 to 0.5 should be for NN
        self._vel_scale = 1 / 0.07 # 2 / 0.14  # maps vel to -1..1

        if log is not None:
            log.add_param('type', 'keras sequential')
            log.add_param('input_count', input_count)
            #log.add_param('hid_1_size', 128)
            #log.add_param('hid_1_act', 'sigmoid')
            log.add_param('output_count', output_count)
            log.add_param('out_act', 'linear')

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
        assert action in list(range(self._output_count))

        return pos, vel, action

    def estimate(self, state, action):
        pos, vel, action = self._test_input(state, action)

        pos += self._pos_offset
        pos *= self._pos_scale
        vel *= self._vel_scale

        est = self._model.predict(np.array([[pos, vel]]))

        assert action in list(range(self._output_count))

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

        states = np.array(states)

        states[:,0] += self._pos_offset
        states[:,0] *= self._pos_scale
        states[:,1] *= self._vel_scale

        est = self._model.predict(states, batch_size=len(states))
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
            est = self._model.predict(inp, batch_size=len(inp))
            est_n = self._model.predict(inp_n, batch_size=len(inp_n))
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
        self._model.train_on_batch(inputs, targets)
        timing_dict['    update_train_on_batch'] += time.time() - time_start

    def update2(self, states, actions, rewards_n, states_n, dones, timing_dict, debug=False):
        assert isinstance(states, np.ndarray)
        assert states.dtype == float
        assert states.ndim == 2
        assert states.shape[0] >= 1
        assert states.shape[1] == self._input_count

        assert isinstance(actions, np.ndarray)
        assert actions.dtype == int
        assert actions.ndim == 2
        assert actions.shape[0] >= 1
        assert actions.shape[1] == 1

        assert isinstance(rewards_n, np.ndarray)
        assert rewards_n.dtype == float
        assert rewards_n.ndim == 2
        assert rewards_n.shape[0] >= 1
        assert rewards_n.shape[1] == 1

        assert isinstance(states_n, np.ndarray)
        assert states_n.dtype == float
        assert states_n.ndim == 2
        assert states_n.shape[0] >= 1
        assert states_n.shape[1] == self._input_count

        assert isinstance(dones, np.ndarray)
        assert dones.dtype == bool
        assert dones.ndim == 2
        assert dones.shape[0] >= 1
        assert dones.shape[1] == 1



        # assert len(batch) > 0
        # assert len(batch[0]) == 5

        # time_start = time.time()
        # inputs = np.zeros([len(batch), 2], dtype=np.float32)
        # actions = np.zeros([len(batch)], dtype=np.int8)
        # rewards_n = np.zeros([len(batch), 1], dtype=np.float32)
        # inputs_n = np.zeros([len(batch), 2], dtype=np.float32)
        # not_dones = np.zeros([len(batch), 1], dtype=np.bool)
        # timing_dict['    update2_create_arr'] += time.time() - time_start

        time_start = time.time()
        inputs = np.array(states)
        inputs_n = np.array(states_n)
        not_dones = np.logical_not(dones)
        timing_dict['    update2_create_arr'] += time.time() - time_start

        # time_start = time.time()
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
        # timing_dict['    update2_loop'] += time.time() - time_start

        time_start = time.time()
        inputs[:,0] += self._pos_offset
        inputs[:,0] *= self._pos_scale
        inputs[:,1] *= self._vel_scale
        inputs_n[:,0] += self._pos_offset
        inputs_n[:,0] *= self._pos_scale
        inputs_n[:,1] *= self._vel_scale
        timing_dict['    update2_scale'] += time.time() - time_start

        time_start = time.time()
        # Join arrays for single predict() call (double speed improvement)
        inputs_joint = np.concatenate((inputs, inputs_n))
        result_joint = self._model.predict(inputs_joint, batch_size=len(inputs_joint))
        targets, est_n = np.split(result_joint, 2)
        timing_dict['    update2_predict'] += time.time() - time_start

        time_start = time.time()
        q_n = np.max(est_n, axis=1, keepdims=True)
        tt = rewards_n + (not_dones * self._discount * q_n)
        errors = tt.flatten() - targets[np.arange(len(targets)), actions.flatten()]
        targets[np.arange(len(targets)), actions.flatten()] = tt.flatten()
        timing_dict['    update2_post'] += time.time() - time_start

        if debug:
            print('inputs')
            print(inputs[0:10])
            print(targets[0:10])
            print(self._model.layers[0].get_weights()[0][:,0:10])
            print(np.sum(np.sum(self._model.layers[0].get_weights())))
            exit()
            # print('TRAIN:')
            # for i in range(len(states)):
            #     print(i, states[i], actions[i], rewards_n[i], states_n[i])
            #     print(i, inputs[i], targets[i])

        time_start = time.time()
        #self._model.train_on_batch(inputs, targets)
        self._model.fit(inputs, targets, batch_size=self._batch_size, epochs=1, verbose=False)
        timing_dict['    update2_train_on_batch'] += time.time() - time_start

        return errors


class HistoryData:
    """One piece of agent trajectory"""
    def __init__(self, observation, reward, done):
        assert isinstance(observation, np.ndarray)
        self.observation = observation
        self.reward = reward
        self.action = None
        self.done = done

    def __str__(self):
        return 'obs={0}, rew={1} done={2}   act={3}'.format(
            self.observation, self.reward, self.done, self.action)


class Agent:
    def __init__(self,
        nb_actions,
        discount,
        nb_rand_steps,
        e_rand_start,
        e_rand_target,
        e_rand_decay,

        mem_size_max,
        mem_enable_pmr,

        approximator,
        step_size,
        batch_size,
        log_agent=None, log_q_val=None, log_hist=None, 
        log_memory=None, log_approx=None,

        seed=None):

        self._random = random.Random()
        if seed is not None:
            self._random.seed(seed)

        self._nb_actions = nb_actions
        self._action_space = list(range(nb_actions))

        # usually gamma in literature
        self._discount = discount

        # if true, exec random action until memory is full
        self._nb_rand_steps = nb_rand_steps  

        # policy parameter, 0 => always greedy
        self._epsilon_random = e_rand_start
        self._epsilon_random_start = e_rand_start
        self._epsilon_random_target = e_rand_target
        self._epsilon_random_decay = e_rand_decay

        self._this_step_rand_act = False

        if approximator == 'aggregate':
            self.Q = AggregateApproximator(
                step_size, self._action_space, init_val=0, log=log_approx)
        elif approximator == 'tile':
            self.Q = TileApproximator(
                step_size, self._action_space, init_val=0, log=log_approx)
        elif approximator == 'neural':
            self.Q = NeuralApproximator(
                step_size, discount, batch_size, log=log_approx)
        elif approximator == 'keras':
            self.Q = KerasApproximator( 2, nb_actions,
                step_size, discount, batch_size, log=log_approx)
        else:
            raise ValueError('Unknown approximator')

        self._memory = memory.Memory(
            state_shape=(2, ),
            act_shape=(1, ),
            dtypes=(float, int, float, float, bool, float),
            max_len=mem_size_max,
            enable_pmr=mem_enable_pmr,
            initial_pmr_error=1000.0,
            seed=seed)

        self._step_size = step_size  # usually noted as alpha in literature
        self._batch_size = batch_size

        

        self._episode = 0
        self._trajectory = []        # Agent saves history on it's way
                                     # this resets every new episode

        self._force_random_action = False

        self._curr_total_step = 0
        self._curr_non_rand_step = 0

        self._debug_cum_state = 0
        self._debug_cum_action = 0
        self._debug_cum_reward = 0
        self._debug_cum_done = 0


        self.log_agent = log_agent
        if log_agent is not None:
            log_agent.add_param('discount', self._discount)
            log_agent.add_param('nb_rand_steps', self._nb_rand_steps)
            
            log_agent.add_param('e_rand_start', self._epsilon_random_start)
            log_agent.add_param('e_rand_target', self._epsilon_random_target)
            log_agent.add_param('e_rand_decay', self._epsilon_random_decay)

            log_agent.add_param('step_size', self._step_size)
            log_agent.add_param('batch_size', self._batch_size)

            log_agent.add_data_item('e_rand')
            log_agent.add_data_item('rand_act')

        self.log_q_val = log_q_val
        if log_q_val is not None:
            log_q_val.add_data_item('q_val')
            log_q_val.add_data_item('series_E0') # Q at point [0.4, 0.035]
            log_q_val.add_data_item('series_E1')
            log_q_val.add_data_item('series_E2')

        self.log_hist = log_hist
        if log_hist is not None:
            log_hist.add_data_item('Rt')
            log_hist.add_data_item('St_pos')
            log_hist.add_data_item('St_vel')
            log_hist.add_data_item('At')
            log_hist.add_data_item('done')

        self.log_memory = log_memory
        if log_memory is not None:
            log_memory.add_param('max_size', mem_size_max)
            log_memory.add_param('enable_pmr', mem_enable_pmr)
            log_memory.add_data_item('curr_size')
            log_memory.add_data_item('hist_St')
            log_memory.add_data_item('hist_At')
            log_memory.add_data_item('hist_Rt_1')
            log_memory.add_data_item('hist_St_1')
            log_memory.add_data_item('hist_done')
            log_memory.add_data_item('hist_error')

    def get_fingerprint(self):
        weights_sum = self.Q.get_weights_fingerprint()

        fingerprint = weights_sum + self._debug_cum_state \
                      + self._debug_cum_action + self._debug_cum_reward \
                      + self._debug_cum_done

        return fingerprint, weights_sum, self._debug_cum_state, \
                self._debug_cum_action, self._debug_cum_reward, \
                self._debug_cum_done

    def reset(self, expl_start=False):
        self._episode += 1
        self._trajectory = []        # Agent saves history on it's way

        self._force_random_action = expl_start

    def log(self, episode, step, total_step):

        #
        #   Log agent
        #
        self.log_agent.append(episode, step, total_step,
            e_rand=self._epsilon_random,
            rand_act=self._this_step_rand_act)

        #
        #   Log history
        #
        self.log_hist.append(episode, step, total_step,
            Rt=self._trajectory[-1].reward,
            St_pos=self._trajectory[-1].observation[0],
            St_vel=self._trajectory[-1].observation[1],
            At=self._trajectory[-1].action,
            done=self._trajectory[-1].done)

        #
        #   Log Q values
        #
        if total_step % 1000 == 0:
            positions = np.linspace(-1.2, 0.5, 64)
            velocities = np.linspace(-0.07, 0.07, 64)

            num_tests = len(positions) * len(velocities)
            pi_skip = len(velocities)
            states = np.zeros([num_tests, 2])
            for pi in range(len(positions)):
                for vi in range(len(velocities)):
                    states[pi*pi_skip + vi, 0] = positions[pi]
                    states[pi*pi_skip + vi, 1] = velocities[vi]


            q_list = self.Q.estimate_all(states)
            q_val = np.zeros([len(positions), len(velocities), self._nb_actions])
            
            for si in range(len(states)):    
                pi = si//pi_skip
                vi = si %pi_skip
                q_val[pi, vi] = q_list[si]

        else:
            q_val=None


        #
        #   Log Memory
        #
        if total_step % 10000 == 0:
            ptr = self._memory._curr_insert_ptr
            # print('ptr', ptr)
            # aa = self._memory._hist_St[ptr:]
            # bb = self._memory._hist_St[0:ptr]
            # cc = np.concatenate((aa, bb))
            self.log_memory.append(episode, step, total_step,
                curr_size=self._memory.length(),
                hist_St=np.concatenate((self._memory._hist_St[ptr:], self._memory._hist_St[0:ptr])),
                hist_At=np.concatenate((self._memory._hist_At[ptr:], self._memory._hist_At[0:ptr])),
                hist_Rt_1=np.concatenate((self._memory._hist_Rt_1[ptr:], self._memory._hist_Rt_1[0:ptr])),
                hist_St_1=np.concatenate((self._memory._hist_St_1[ptr:], self._memory._hist_St_1[0:ptr])),
                hist_done=np.concatenate((self._memory._hist_done[ptr:], self._memory._hist_done[0:ptr])),
                hist_error=np.concatenate((self._memory._hist_error[ptr:], self._memory._hist_error[0:ptr])) )
        else:
            self.log_memory.append(episode, step, total_step,
                curr_size=None,
                hist_St=None,
                hist_At=None,
                hist_Rt_1=None,
                hist_St_1=None,
                hist_done=None,
                hist_error=None
                )
        
        #
        #   Log Q series
        #
        if total_step % 100 == 0:
            est = self.Q.estimate_all(np.array([[0.4, 0.035]]))
        else:
            est = np.array([[None, None, None]])

        self.log_q_val.append(episode, step, total_step,
            q_val=q_val,
            series_E0=est[0, 0], series_E1=est[0, 1], series_E2=None)#est[0, 2])

    def advance_one_step(self):
        self._curr_total_step += 1

        if self._curr_total_step > self._nb_rand_steps:
            self._curr_non_rand_step += 1

            #
            #   Decrease linearly
            #
            if self._epsilon_random > self._epsilon_random_target:
                self._epsilon_random -= self._epsilon_random_decay
            if self._epsilon_random < self._epsilon_random_target:
                self._epsilon_random = self._epsilon_random_target
            
            #
            #   Decrease exponentially
            #
            # self._epsilon_random = \
            #     self._epsilon_random_target \
            #     + (self._epsilon_random_start - self._epsilon_random_target) \
            #     * math.exp(-self._epsilon_random_decay * self._curr_non_rand_step)

    def pick_action(self, obs):
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (2, )

        if self._curr_total_step < self._nb_rand_steps:
            self._this_step_rand_act = True
            # return np.random.choice(self._action_space)
            result = self._random.randint(0, self._nb_actions-1)
            return result

        if self._force_random_action:
            self._force_random_action = False
            self._this_step_rand_act = True
            return np.random.choice(self._action_space)

        #if np.random.rand() < self._epsilon_random:
        if self._random.random() < self._epsilon_random:
            # pick random action
            self._this_step_rand_act = True
            # res = np.random.choice(self._action_space)
            res = self._random.randint(0, self._nb_actions-1)

        else:
            self._this_step_rand_act = False
            # act greedy
            
            # max_Q = float('-inf')
            # max_action = None
            # possible_actions = []
            # for action in self._action_space:
            #     q = self.Q.estimate(obs, action)
            #     if q > max_Q:
            #         possible_actions.clear()
            #         possible_actions.append(action)
            #         max_Q = q
            #     elif q == max_Q:
            #         possible_actions.append(action)
            # res = np.random.choice(possible_actions)

            obs = obs.reshape([1, 2])
            q_arr = self.Q.estimate_all(obs).flatten()
            index = _rand_argmax(q_arr)
            res = self._action_space[index]

        return res



    def append_trajectory(self, observation, reward, done):
        self._debug_cum_state += np.sum(observation)

        if reward is not None:
            self._debug_cum_reward += reward
        if done is not None:
            self._debug_cum_done += int(done)

        self._trajectory.append(
            HistoryData(observation, reward, done))

    def append_action(self, action):
        self._debug_cum_action += np.sum(action)

        if len(self._trajectory) != 0:
            self._trajectory[-1].action = action

    def print_trajectory(self):
        print('Trajectory:')
        for element in self._trajectory:
            print(element)
        print('Total trajectory steps: {0}'.format(len(self._trajectory)))

    def check_trajectory_terminated_ok(self):
        last_entry = self._trajectory[-1]
        if not last_entry.done:
            raise ValueError('Cant do offline on non-terminated episode')
        # for act in self._action_space:
        #     if self.Q[last_entry.observation, act] != 0:
        #         raise ValueError('Action from last state has non-zero val.')



    def eval_td_t(self, t, timing_dict):
        """TD update state-value for single state in trajectory

        This assumesss time step t+1 is availalbe in the trajectory

        For online updates:
            Call with t equal to previous time step

        For offline updates:
            Iterate trajectory from t=0 to t=T-1 and call for every t

        Params:
            t (int [t, T-1]) - time step in trajectory,
                    0 is initial state; T-1 is last non-terminal state

        """

        time_start = time.time()

        # Shortcuts for more compact notation:

        St = self._trajectory[t].observation      # evaluated state tuple (x, y)
        At = self._trajectory[t].action
        St_1 = self._trajectory[t+1].observation  # next state tuple (x, y)
        Rt_1 = self._trajectory[t+1].reward       # next step reward
        done = self._trajectory[t+1].done
        self._memory.append(St, At, Rt_1, St_1, done)

        if self._curr_total_step < self._nb_rand_steps:
            # no lerninng during initial random phase
            timing_dict['  eval_td_start'] += time.time() - time_start
            return

        timing_dict['  eval_td_start'] += time.time() - time_start


        if isinstance(self.Q, NeuralApproximator) or \
            isinstance(self.Q, KerasApproximator):

            time_start = time.time()
            states, actions, rewards_1, states_1, dones, indices = \
                self._memory.get_batch(self._batch_size)
            timing_dict['  eval_td_get_batch'] += time.time() - time_start

            time_start = time.time()
            debug = self._curr_total_step == 110500
            errors = self.Q.update2(states, actions, rewards_1, states_1, dones, timing_dict)
            timing_dict['  eval_td_update'] += time.time() - time_start

            self._memory.update_errors(indices, errors)

        else:
            if done:
                Tt = Rt_1
            else:
                At_1 = self._trajectory[t+1].action
                if At_1 is None:
                    At_1 = self.pick_action(St)
                Tt = Rt_1 + self._discount * self.Q.estimate(St_1, At_1)                

            self.Q.update(St, At, Tt)
            

    def eval_td_online(self, timing_dict):
        self.eval_td_t(len(self._trajectory) - 2, timing_dict)  # Eval next-to last state

