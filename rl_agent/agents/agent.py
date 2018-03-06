import numpy as np

from .agent_base import AgentBase
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


    def reset(self):
        """Reset after episode completed. Can be called manually if required"""
        super().reset()
        self.policy.reset()


    def next_step(self, done):
        super().next_step(done)
        self.policy.next_step(self._curr_total_step)


    def log(self, episode, step, total_step):
        super().log(episode, step, total_step)
        if self.memory is not None:
            self.memory.log(episode, step, total_step)
        if self.Q is not None:
            self.Q.log(episode, step, total_step)


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
            

    def get_fingerprint(self):
        weights_sum = self.Q.get_weights_fingerprint()

        fingerprint = weights_sum \
                      + self._debug_cum_state \
                      + self._debug_cum_action \
                      + self._debug_cum_reward \
                      + self._debug_cum_done

        return fingerprint, weights_sum, self._debug_cum_state, \
                self._debug_cum_action, self._debug_cum_reward, \
                self._debug_cum_done
