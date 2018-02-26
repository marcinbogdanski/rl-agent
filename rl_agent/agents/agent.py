import numpy as np
import matplotlib.pyplot as plt
import collections
import pdb
import random
import math


from . import memory
from .approximators import AggregateApproximator
from .approximators import TilesApproximator
from .approximators import NeuralApproximator
from .approximators import KerasApproximator

import tensorflow as tf



def _rand_argmax(vector):
    """ Argmax that chooses randomly among eligible maximum indices. """
    vmax = np.max(vector)
    indices = np.nonzero(vector == vmax)[0]
    return np.random.choice(indices)



        







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
        expl_start,
        nb_rand_steps,
        e_rand_start,
        e_rand_target,
        e_rand_decay,

        mem_size_max,
        mem_enable_pmr,

        approximator,
        step_size,
        batch_size,
        logger=None,

        seed=None):

        self._random = random.Random()
        if seed is not None:
            self._random.seed(seed)

        self._nb_actions = nb_actions
        self._action_space = list(range(nb_actions))

        # usually gamma in literature
        self._discount = discount

        # if set, first action in episode will always be random
        self._expl_start = expl_start

        # if true, exec random action until memory is full
        self.nb_rand_steps = nb_rand_steps  

        # policy parameter, 0 => always greedy
        self._epsilon_random = e_rand_start
        self._epsilon_random_start = e_rand_start
        self._epsilon_random_target = e_rand_target
        self._epsilon_random_decay = e_rand_decay

        self._this_step_rand_act = False

        log_approx = logger.approx if logger is not None else None

        if approximator == 'aggregate':
            self.Q = AggregateApproximator(
                step_size, self._action_space, init_val=0, log=log_approx)
        elif approximator == 'tiles':
            self.Q = TilesApproximator(
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

        

        self._completed_episodes = 0
        self._trajectory = []        # Agent saves history on it's way
                                     # this resets every new episode

        self._force_random_action = False

        self._curr_total_step = 0
        self._curr_non_rand_step = 0

        self._debug_cum_state = 0
        self._debug_cum_action = 0
        self._debug_cum_reward = 0
        self._debug_cum_done = 0

        self.logger = logger
        if self.logger is not None:
            self.logger.agent.add_param('discount', self._discount)
            self.logger.agent.add_param('nb_rand_steps', self.nb_rand_steps)
            
            self.logger.agent.add_param('e_rand_start', self._epsilon_random_start)
            self.logger.agent.add_param('e_rand_target', self._epsilon_random_target)
            self.logger.agent.add_param('e_rand_decay', self._epsilon_random_decay)

            self.logger.agent.add_param('step_size', self._step_size)
            self.logger.agent.add_param('batch_size', self._batch_size)

            self.logger.agent.add_data_item('e_rand')
            self.logger.agent.add_data_item('rand_act')

        if self.logger is not None:
            self.logger.q_val.add_data_item('q_val')
            self.logger.q_val.add_data_item('series_E0') # Q at point [0.4, 0.035]
            self.logger.q_val.add_data_item('series_E1')
            self.logger.q_val.add_data_item('series_E2')

        if self.logger is not None:
            self.logger.hist.add_data_item('Rt')
            self.logger.hist.add_data_item('St_pos')
            self.logger.hist.add_data_item('St_vel')
            self.logger.hist.add_data_item('At')
            self.logger.hist.add_data_item('done')

        if self.logger is not None:
            self.logger.memory.add_param('max_size', mem_size_max)
            self.logger.memory.add_param('enable_pmr', mem_enable_pmr)
            self.logger.memory.add_data_item('curr_size')
            self.logger.memory.add_data_item('hist_St')
            self.logger.memory.add_data_item('hist_At')
            self.logger.memory.add_data_item('hist_Rt_1')
            self.logger.memory.add_data_item('hist_St_1')
            self.logger.memory.add_data_item('hist_done')
            self.logger.memory.add_data_item('hist_error')

    @property
    def step(self):
        return self._curr_step

    @property
    def total_step(self):
        return self._curr_total_step

    @property
    def completed_episodes(self):
        return self._completed_episodes

    def get_fingerprint(self):
        weights_sum = self.Q.get_weights_fingerprint()

        fingerprint = weights_sum + self._debug_cum_state \
                      + self._debug_cum_action + self._debug_cum_reward \
                      + self._debug_cum_done

        return fingerprint, weights_sum, self._debug_cum_state, \
                self._debug_cum_action, self._debug_cum_reward, \
                self._debug_cum_done

    def reset(self):
        self._curr_step = 0
        self._trajectory = []        # Agent saves history on it's way

        self._force_random_action = self._expl_start

    def log(self, episode, step, total_step):
        
        if self.logger is None:
            return

        #
        #   Log agent
        #
        self.logger.agent.append(episode, step, total_step,
            e_rand=self._epsilon_random,
            rand_act=self._this_step_rand_act)

        #
        #   Log history
        #
        self.logger.hist.append(episode, step, total_step,
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
            self.logger.memory.append(episode, step, total_step,
                curr_size=self._memory.length(),
                hist_St=np.concatenate((self._memory._hist_St[ptr:], self._memory._hist_St[0:ptr])),
                hist_At=np.concatenate((self._memory._hist_At[ptr:], self._memory._hist_At[0:ptr])),
                hist_Rt_1=np.concatenate((self._memory._hist_Rt_1[ptr:], self._memory._hist_Rt_1[0:ptr])),
                hist_St_1=np.concatenate((self._memory._hist_St_1[ptr:], self._memory._hist_St_1[0:ptr])),
                hist_done=np.concatenate((self._memory._hist_done[ptr:], self._memory._hist_done[0:ptr])),
                hist_error=np.concatenate((self._memory._hist_error[ptr:], self._memory._hist_error[0:ptr])) )
        else:
            self.logger.memory.append(episode, step, total_step,
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

        self.logger.q_val.append(episode, step, total_step,
            q_val=q_val,
            series_E0=est[0, 0], series_E1=est[0, 1], series_E2=None)#est[0, 2])

    def advance_one_step(self):
        self._curr_step += 1
        self._curr_total_step += 1

        if self._trajectory[-1].done is True:
            self._completed_episodes += 1

        if self._curr_total_step > self.nb_rand_steps:
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

        if self._curr_total_step < self.nb_rand_steps:
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


    def learn(self):
        self.eval_td_online()

    def eval_td_online(self):
        self.eval_td_t(len(self._trajectory) - 2)  # Eval next-to last state

    def eval_td_t(self, t):
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

        # Shortcuts for more compact notation:

        St = self._trajectory[t].observation      # evaluated state tuple (x, y)
        At = self._trajectory[t].action
        St_1 = self._trajectory[t+1].observation  # next state tuple (x, y)
        Rt_1 = self._trajectory[t+1].reward       # next step reward
        done = self._trajectory[t+1].done
        self._memory.append(St, At, Rt_1, St_1, done)

        if self._curr_total_step < self.nb_rand_steps:
            # no lerninng during initial random phase
            return



        if isinstance(self.Q, NeuralApproximator) or \
            isinstance(self.Q, KerasApproximator):

            states, actions, rewards_1, states_1, dones, indices = \
                self._memory.get_batch(self._batch_size)

            debug = self._curr_total_step == 110500
            errors = self.Q.update2(states, actions, rewards_1, states_1, dones)

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
            

