import numpy as np
import tensorflow as tf

from . import memory
from .approximators import KerasApproximator


import pdb


def _rand_argmax(vector):
    """ Argmax that chooses randomly among eligible maximum indices. """
    vmax = np.max(vector)
    indices = np.nonzero(vector == vmax)[0]
    return np.random.choice(indices)


class HistoryData:
    """One piece of agent trajectory (obs, rew, act, done)"""
    def __init__(self, total_step, observation, reward, done):
        assert isinstance(total_step, int)
        assert total_step >= 0
        assert isinstance(observation, np.ndarray)
        self.total_step = total_step
        self.observation = observation
        self.reward = reward
        self.action = None
        self.done = done

    def __str__(self):
        return 'obs={0}, rew={1} done={2}   act={3}'.format(
            self.observation, self.reward, self.done, self.action)


class EpisodeData:
    """Episode summary"""
    def __init__(self, start, end, length, total_reward):
        assert end-start == length-1
        self.start = start
        self.end = end
        self.length = length
        self.total_reward = total_reward


class Agent:
    def __init__(self,
        state_space,
        action_space,
        discount,
        start_learning_at,

        mem_size_max,
        mem_batch_size,
        mem_enable_pmr,

        q_fun_approx,
        policy):

        self._state_space = state_space
        self._action_space = action_space

        # usually gamma in literature
        self._discount = discount
        self._start_learning_at = start_learning_at


        #
        #   Initialize Q-function approximator
        #
        self.Q = q_fun_approx
        if self.Q is not None:
            self.Q.set_state_action_spaces(state_space, action_space)

        #
        #   Initialise Memory
        #
        self.memory = memory.Memory(
            state_space=state_space,
            action_space=action_space,
            max_len=mem_size_max,
            enable_pmr=mem_enable_pmr,
            initial_pmr_error=1000.0)

        self._mem_batch_size = mem_batch_size

        #
        #   Initialize Policy
        #
        self.policy = policy
        self.policy.set_state_action_spaces(state_space, action_space)
        self.policy.link(self)  # Policy may need access to Q-approx etc.


        self._this_step_rand_act = False

        

        self._completed_episodes = 0

        self._callback_on_step_end = None
        self._callback_on_step_end_params = None
        
        self._episodes_history = []
        self._trajectory = []
        self.reset()

        self._curr_total_step = 0

        self._debug_cum_state = 0
        self._debug_cum_action = 0
        self._debug_cum_reward = 0
        self._debug_cum_done = 0

        self.log_episodes = None
        self.log_hist = None

    @property
    def start_learning_at(self):
        return self._start_learning_at

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

    def get_avg_ep_reward(self, nb_episodes):
        hist_chunk = self._episodes_history[-nb_episodes:]
        if len(hist_chunk) == 0:
            return None
        else:
            sum_ = sum(ep_hist.total_reward for ep_hist in hist_chunk)
            return sum_ / len(hist_chunk)

    def get_cont_reward(self, nb_steps):
        if len(self._trajectory) <= nb_steps:
            return None

        hist_chunk = self._trajectory[-nb_steps:]
        return sum(x.reward for x in hist_chunk if x.reward)

    def reset(self):
        if len(self._trajectory) > 0:
            # save summary into episode history
            ep_start = self._trajectory[0].total_step
            ep_end = self._trajectory[-1].total_step
            ep_len = len(self._trajectory)

            total_reward = 0
            for i in range(len(self._trajectory)):
                if self._trajectory[i].reward is not None:
                    total_reward += self._trajectory[i].reward

            ep_hist = EpisodeData(ep_start, ep_end, ep_len, total_reward)
            self._episodes_history.append(ep_hist)

            self._completed_episodes += 1

            if self.log_episodes is not None:
                self.log_episodes.append(
                    self.completed_episodes, self.step, self.total_step,
                    start=ep_start, end=ep_end, reward=total_reward)           

        self._curr_step = 0
        self._trajectory = []        # Agent saves history on it's way

        self.policy.reset()

    def log(self, episode, step, total_step):
        
        #
        #   Log episodes
        #
        if self.log_episodes is not None \
            and not self.log_episodes.is_initialized:

            self.log_episodes.add_data_item('start')
            self.log_episodes.add_data_item('end')
            self.log_episodes.add_data_item('reward')

        #
        #   Log history
        #
        if self.log_hist is not None and not self.log_hist.is_initialized:
            self.log_hist.add_data_item('Rt')
            self.log_hist.add_data_item('St_pos')
            self.log_hist.add_data_item('St_vel')
            self.log_hist.add_data_item('At')
            self.log_hist.add_data_item('done')

        if self.log_hist is not None:
            self.log_hist.append(episode, step, total_step,
                Rt=self._trajectory[-1].reward,
                St_pos=self._trajectory[-1].observation[0],
                St_vel=self._trajectory[-1].observation[1],
                At=self._trajectory[-1].action,
                done=self._trajectory[-1].done)

        self.memory.log(episode, step, total_step)
        self.Q.log(episode, step, total_step)


    def register_callback(self, which, function):

        if which == 'on_step_end':
            if self._callback_on_step_end is not None:
                raise ValueError('callback {} already registered')
            self._callback_on_step_end = function
        else:
            raise ValueError('unknown callback routine' + which)

    def clear_callback(self, which):
        if which == 'on_step_end':
            self._callback_on_step_end = None
        else:
            raise ValueError('unknown callback routine' + which)


    def next_step(self, done):

        self.log(self.completed_episodes, self.step, self.total_step)

        if self._callback_on_step_end is not None:
            self._callback_on_step_end(
                agent=self,
                reward=self._trajectory[-1].reward,
                observation=self._trajectory[-1].observation,
                done=self._trajectory[-1].done,
                action=self._trajectory[-1].action)

        # -- roll into next time step --

        self._curr_step += 1
        self._curr_total_step += 1

        if done:
            self.reset()

        self.policy.next_step(self._curr_total_step)
 

    def take_action(self, obs):
        if self._trajectory[-1].done is True:
            return None
        else:
            action = self.policy.pick_action(obs)
            self._append_action(action)
            return action


    def _append_action(self, action):
        assert len(self._trajectory) != 0
        self._debug_cum_action += np.sum(action)
        self._trajectory[-1].action = action


    def observe(self, observation, reward, done):
        self._debug_cum_state += np.sum(observation)

        if reward is not None:
            self._debug_cum_reward += reward
        if done is not None:
            self._debug_cum_done += int(done)

        self._trajectory.append(
            HistoryData(self._curr_total_step, observation, reward, done))


    def print_trajectory(self):
        print('Trajectory:')
        for element in self._trajectory:
            print(element)
        print('Total trajectory steps: {0}'.format(len(self._trajectory)))


    def check_trajectory_terminated_ok(self):
        last_entry = self._trajectory[-1]
        if not last_entry.done:
            raise ValueError('Cant do offline on non-terminated episode')


    def learn(self):
        self.eval_td_online()


    def eval_td_online(self):
        if len(self._trajectory) >= 2:
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
        self.memory.append(St, At, Rt_1, St_1, done)

        if self._curr_total_step < self._start_learning_at:
            # no lerninng during initial random phase
            return


        if isinstance(self.Q, KerasApproximator):

            #
            #   Q-Learning
            #

            # Get batch
            states, actions, rewards_1, states_1, dones, indices = \
                self.memory.get_batch(self._mem_batch_size)
            
            # Calculates max_Q for next states
            q_max = self.Q.max_op(states_1)

            # True if state non-terminal
            not_dones = np.logical_not(dones)

            # Calc target Q values (reward_t + discout * max(Q_t_plus_1))
            targets = rewards_1 + (not_dones * self._discount * q_max)

            errors = self.Q.train(states, actions, targets)

            self.memory.update_errors(indices, np.abs(errors))

        else:

            #
            #   SARSA
            #
            if done:
                Tt = Rt_1
            else:
                At_1 = self._trajectory[t+1].action
                if At_1 is None:
                    # TODO: should this be St, or St_1 !?!
                    At_1 = self.policy.pick_action(St_1)
                Tt = Rt_1 + self._discount * self.Q.estimate(St_1, At_1)                

            self.Q.train(St, At, Tt)
            

