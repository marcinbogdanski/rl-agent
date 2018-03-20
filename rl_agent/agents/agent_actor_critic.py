import numpy as np

from .agent_base import AgentBase
from .memory_basic import MemoryBasic

import pdb

# TODO: doc
# TODO: unit test

class AgentActorCritic(AgentBase):
    """Watis until episode end, then learns on all states/actions visited.

    See base class doc for how to use Agents in general.
    """

    def __init__(self,
        state_space,
        action_space,
        discount,

        algorithm,
        nb_episodes_in_batch,

        memory,
        v_fun_approx,
        q_fun_approx,
        policy):
        """
        Params:
            state_space: gym.spaces object describing state space
            action_space: gym.spaces object describing action space
            discount: per-step reward discount, usually gamma in literature
        """
        super().__init__(state_space, action_space, discount)

        assert algorithm in ['raw', 'mc_return']
        assert isinstance(nb_episodes_in_batch, int)
        assert nb_episodes_in_batch >= 1
        

        self._algorithm = algorithm
        self._nb_episodes_in_batch = nb_episodes_in_batch

        #
        #   Initialize Memory Module
        #
        if not isinstance(memory, MemoryBasic):
            raise ValueError('Memory must be of type MemoryBasic')
        self.memory = memory
        self.memory.set_state_action_spaces(state_space, action_space)

        #
        #   Initialize Q-function approximator
        #
        self.V = v_fun_approx
        if self.V is not None:
            self.V.set_state_action_spaces(state_space, action_space)

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


    def perform_logging(self, episode, step, total_step):
        super().perform_logging(episode, step, total_step)
        if self.Q is not None:
            self.Q.perform_logging(episode, step, total_step)


    def observe(self, observation, reward, done):
        super().observe(observation, reward, done)

        if len(self._trajectory) >= 2:
            t = len(self._trajectory)-2               # next to last step
            St = self._trajectory[t].observation
            At = self._trajectory[t].action
            St_1 = self._trajectory[t+1].observation  # next state
            Rt_1 = self._trajectory[t+1].reward       # next step reward
            done = self._trajectory[t+1].done         # next step done flag
            self.memory.append(St, At, Rt_1, St_1, done)


    def learn(self):
        """Perform MC update on completed episodes"""

        if self._algorithm == 'mc_return':
            self._learn_mc_return()


    def _learn_mc_return(self):

        if self.memory.nb_episodes == self._nb_episodes_in_batch:

            assert self._trajectory[-1].done

            # Iterate all states in trajectory, apart from terminal state
            for t in range(0, len(self._trajectory)-1):
                # Update state-value at time t
                self._eval_mc_t(t)

            self.memory.clear()

        # # Do MC update only if episode terminated
        # if self._trajectory[-1].done:

        #     # Iterate all states in trajectory, apart from terminal state
        #     for t in range(0, len(self._trajectory)-1):
        #         # Update state-value at time t
        #         self._eval_mc_t(t)

    def _eval_mc_t(self, t):
        """MC update for state-values for single state in trajectory

        Note:
            This assumes episode is completed and trajectory is present
            from start to termination.

        Params:
            t (int [t, T-1]) - time step in trajectory,
                    0 is initial state; T-1 is last non-terminal state
        """


        # Shortcuts for more compact notation:
        St = self._trajectory[t].observation  # current state
        At = self._trajectory[t].action
        Gt = self._calc_Gt(t)                  # return for current state

        if self.Q is not None:
            self.Q.train(St, At, Gt)

        self.policy.train_single(St, At, Gt)


    def _calc_Gt(self, t, n=float('inf')):
        """Calculates return for state t, using n future steps.

        Params:
            t (int [t, T-1]) - time step in trajectory,
                    0 is initial state; T-1 is last non-terminal state

            n (int or +inf, [0, +inf]) - n-steps of reward to accumulate
                    If n >= T then calculate full return for state t
                    For n == 1 this equals to TD return
                    For n == +inf this equals to MC return
        """

        
        discount = 1.0

        Gt = 0  # return

        T = len(self._trajectory)-1   # terminal state
        for j in range(t+1, T+1):
            Rj = self._trajectory[j].reward
            Gt += discount * Rj
            discount *= self._discount

        return Gt
            

    def get_fingerprint(self):
        weights_sum = 0
        if self.Q is not None:
            weights_sum = self.Q.get_weights_fingerprint()

        fingerprint = weights_sum \
                      + self._debug_cum_state \
                      + self._debug_cum_action \
                      + self._debug_cum_reward \
                      + self._debug_cum_done

        return fingerprint, weights_sum, self._debug_cum_state, \
                self._debug_cum_action, self._debug_cum_reward, \
                self._debug_cum_done
