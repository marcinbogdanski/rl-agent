import numpy as np

from .agent_base import AgentBase, HistoryData, EpisodeData
from .approximators import KerasApproximator


import pdb


class Agent(AgentBase):
    """Generic RL Agent class.

    Can be configured in various ways to match different learning algorithms.

    Timestep convention matches Sutton & Barto 2018:
        *** time step starts ***
        1) either reset env, or perform env.step(action_form_previous_step)
        2) feed obs, reward and done flat to Agent for learning
        3) do Agent.take_action() to select action for next time step
        4) do housekeeping, logging, callbacks etc.
        *** time step ends ***

    Abbreviated example of Agent training loop:
        agent.reset()  # optional, in case agent has non-terminated trajectory
        done = True
        while True:
            # *** time step starts here ***
            if done:                           # reset or step the environment
                obs, reward, done = env.reset(), None, False
            else:
                obs, reward, done, _ = env.step(action)
            agent.observe(obs, reward, done)   # save to internal trajectory
            agent.learn()                      # learn from internal trajectory
            action = agent.take_action(obs)    # pick act and save internally
            agent.next_step()                  # inform agent step is over
            # *** time step ends here ***
            if agent.total_step > train_steps: # check stop conditions
                break

    For more complete example see rl_agent.runner.train_agent() function
    """

    def __init__(self,
        state_space,
        action_space,
        discount,
        start_learning_at,

        memory,
        q_fun_approx,
        policy):
        """
        Params:
            state_space: gym.spaces object describing state space
            action_space: gym.spaces object describing action space
            discount: per-step reward discount, usually gamma in literature
            start_learning_at: postpone any learning until this time step,
                               use e.g. to pre-fill replay memory with random
        """
        super().__init__(
            state_space, action_space,
            discount, start_learning_at)


        #
        #   Initialise Memory Module
        #
        self.memory = memory
        if memory is not None:
            self.memory.set_state_action_spaces(state_space, action_space)

        #
        #   Initialize Q-function approximator
        #
        self.Q = q_fun_approx
        if self.Q is not None:
            self.Q.set_state_action_spaces(state_space, action_space)


        #
        #   Initialize Policy Module
        #
        self.policy = policy
        self.policy.set_state_action_spaces(state_space, action_space)
        self.policy.link(self)  # Policy may need access to Q-approx etc.


        
        #
        #   Housekeeping
        #
        self._curr_total_step = 0
        self._completed_episodes = 0
        self._episodes_history = []
        self._trajectory = []
        self.reset()

        #
        #   Callbacks
        #
        self._callback_on_step_end = None
        self._callback_on_step_end_params = None

        #
        #   Loggers
        #
        self.log_episodes = None  # per-episode logger (episode summary)
        self.log_hist = None      # per-step logger (saves all states visited)
        
        #
        #   Debug stuff for unittests
        #        
        self._debug_cum_state = 0
        self._debug_cum_action = 0
        self._debug_cum_reward = 0
        self._debug_cum_done = 0

        

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
        """Unique nb that should not change between reproducible sessions"""
        weights_sum = self.Q.get_weights_fingerprint()

        fingerprint = weights_sum + self._debug_cum_state \
                      + self._debug_cum_action + self._debug_cum_reward \
                      + self._debug_cum_done

        return fingerprint, weights_sum, self._debug_cum_state, \
                self._debug_cum_action, self._debug_cum_reward, \
                self._debug_cum_done

    def get_avg_ep_reward(self, nb_episodes):
        """Average reward over last nb_episodes episodes"""
        hist_chunk = self._episodes_history[-nb_episodes:]
        if len(hist_chunk) == 0:
            return None
        else:
            sum_ = sum(ep_hist.total_reward for ep_hist in hist_chunk)
            return sum_ / len(hist_chunk)

    def get_cont_reward(self, nb_steps):
        """Average reward over last nb_steps steps"""
        if len(self._trajectory) <= nb_steps:
            return None

        hist_chunk = self._trajectory[-nb_steps:]
        return sum(x.reward for x in hist_chunk if x.reward)

    def reset(self):
        """Reset after episode completed. Can be called manually if required"""
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
        """Log internal stuff, call log() on all configured modules"""
        
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

        #
        #   Log internal modules
        #
        if self.memory is not None:
            self.memory.log(episode, step, total_step)

        if self.Q is not None:
            self.Q.log(episode, step, total_step)


    def register_callback(self, which, function):
        """Register one of the callbacks.

        Currently only one callback can be registered.

        Args:
            which (str): Currently supported callbacks are:
                'on_step_end' - called last thing before rolling into next step
            function: user supplied function to call, signature must be:
                def my_fun(self, agent, reward, observation, done, action):
        """
        if which == 'on_step_end':
            if self._callback_on_step_end is not None:
                raise ValueError('callback {} already registered')
            self._callback_on_step_end = function
        else:
            raise ValueError('unknown callback routine' + which)

    def clear_callback(self, which):
        """Remove callback"""
        if which == 'on_step_end':
            self._callback_on_step_end = None
        else:
            raise ValueError('unknown callback routine' + which)


    def next_step(self, done):
        """Call to indicate time-step is over so agent can do housekeeping"""

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
        """Pick action save it to trajectory. See agent doc for call order.

        If you want to pick action w/o saving into trajectory,
        call agent.policy.pick_action() directly

        Params:
            obs - observation as returned from environment
        """
        if self._trajectory[-1].done is True:
            return None
        else:
            action = self.policy.pick_action(obs)
            self._append_action(action)
            return action


    def _append_action(self, action):
        """This will append action to last-step in trajectory"""
        assert len(self._trajectory) != 0
        self._debug_cum_action += np.sum(action)
        self._trajectory[-1].action = action


    def observe(self, observation, reward, done):
        """Feed env data into agent. See Agent doc for call order.

        Args:
            observation: obs as returned from environment
            reward: reward as returned from env
            done: done flag as returned from env
        """
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


    def learn(self):
        """Perform learning. See Agent doc for call order."""
        self._eval_td_online()


    def _eval_td_online(self):
        if len(self._trajectory) >= 2:
            self._eval_td_t(len(self._trajectory) - 2)  # Eval next-to last state


    def _eval_td_t(self, t):
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

        if self.memory is not None:
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
                self.memory.get_batch()
            
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
            

