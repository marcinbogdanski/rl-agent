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

        algorithm_type,
        algorithm_mode,
        algorithm_ep_in_batch,

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

        assert algorithm_type in ['raw', 'raw_norm', 'mc_return']
        assert algorithm_mode in ['online', 'batch']
        assert isinstance(algorithm_ep_in_batch, int)
        assert algorithm_ep_in_batch >= 1



        if algorithm_type == 'raw':
            if algorithm_mode != 'batch':
                raise ValueError('Alg. "raw" must be run in batch mode.')
        if algorithm_type == 'raw_norm':
            if algorithm_mode != 'batch':
                raise ValueError('Alg. "raw_norm" must be run in batch mode.')
        if algorithm_type == 'mc_return':
            if algorithm_mode != 'batch':
                raise ValueError('Alg. "mc_return" must be run in batch mode.')

        

        self._algorithm_type = algorithm_type
        self._algorithm_mode = algorithm_mode
        self._algorithm_ep_in_batch = algorithm_ep_in_batch

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

        if self._algorithm_mode == 'online':
            self._learn_online()


        elif self._algorithm_mode == 'batch':
            if self.memory.nb_episodes == self._algorithm_ep_in_batch:
                self._learn_batch()

        else:
            raise ValueError('Uknown mode: ' + self._algorithm_mode)

    def _learn_online(self):
        raise NotImplementedError()

    def _learn_batch(self):
        states, actions, rewards_1, states_1, dones_1 = \
            self.memory.get_all()
        targets = np.zeros([len(states)])

        #
        #   RAW
        #
        if self._algorithm_type == 'raw':
            for i in range(len(states)):
                targets[i] = self._calc_Gt_new(i, rewards_1, dones_1)

        #
        #   RAW NORM
        #
        elif self._algorithm_type == 'raw_norm':
            for i in range(len(states)):
                targets[i] = self._calc_Gt_new(i, rewards_1, dones_1)
            mean = np.mean(targets)
            stddev = np.std(targets)
            targets -= mean
            targets /= stddev + 1e-6  # don't divide by zero

        #
        #   MC RETURN
        #
        elif self._algorithm_type == 'mc_return':
            for i in range(len(states)):
                targets[i] = self._calc_Gt_new(i, rewards_1, dones_1)

        else:
            raise ValueError('Uknown alg. type: '+self._algorithm_type)


        self.policy.train_batch(states, actions, targets)

        self.memory.clear()


    


        # if self.memory.nb_episodes == self._algorithm_ep_in_batch:

        #     print('   ---   old way   ---   ')

        #     assert self._trajectory[-1].done

        #     # Iterate all states in trajectory, apart from terminal state
        #     for t in range(0, len(self._trajectory)-1):
        #         # Update state-value at time t
        #         self._eval_mc_t(t)

        #     self.memory.clear()



        # # Do MC update only if episode terminated
        # if self._trajectory[-1].done:

        #     # Iterate all states in trajectory, apart from terminal state
        #     for t in range(0, len(self._trajectory)-1):
        #         # Update state-value at time t
        #         self._eval_mc_t(t)


    def _calc_Gt_new(self, t, rewards_1, dones_1):
        discount = 1.0

        Gt = 0  # return

        while True:
            Gt += discount * rewards_1[t]
            discount *= self._discount
            if dones_1[t]:
                break
            else:
                t += 1

        return Gt


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

        print('Gt_old', Gt)

        if self.Q is not None:
            self.Q.train(St, At, Gt)

        self.policy.train_single(St, At, Gt)


    def _calc_Gt(self, t):
        """Calculates return for state t, using n future steps.

        Params:
            t (int [t, T-1]) - time step in trajectory,
                    0 is initial state; T-1 is last non-terminal state
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
