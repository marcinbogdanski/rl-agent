import numpy as np
import gym

class QMaxPolicy:
    def __init__(self, expl_start, nb_rand_steps,
            e_rand_start, e_rand_target, e_rand_decay):
        
        # if set, first action in episode will always be random
        self._expl_start = expl_start

        # if true, exec random action until memory is full
        self._nb_rand_steps = nb_rand_steps  

        # policy parameter, 0 => always greedy
        self._epsilon_random = e_rand_start
        self._epsilon_random_start = e_rand_start
        self._epsilon_random_target = e_rand_target
        self._epsilon_random_decay = e_rand_decay

        self._this_step_rand_act = False

        self._state_space = None
        self._action_space = None
        self._q_approximator = None

    @property
    def nb_rand_steps(self):
        return self._nb_rand_steps

    def set_state_action_spaces(self, state_space, action_space):
        # These should be relaxed in the future,
        # possibly remove gym dependancy
        if not isinstance(state_space, gym.spaces.Box):
            raise ValueError('Only gym.spaces.Box state space supproted')
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError('Only gym.spaces.Discrete action space supported')

        self._state_space = state_space
        self._action_space = action_space

    def link(self, agent):
        self._q_approximator = agent.Q

    def reset(self):
        self._force_random_action = self._expl_start

    def next_step(self, total_step):
        if total_step > self.nb_rand_steps:

            #
            #   Decrease linearly
            #
            if self._epsilon_random > self._epsilon_random_target:
                self._epsilon_random -= self._epsilon_random_decay
            if self._epsilon_random < self._epsilon_random_target:
                self._epsilon_random = self._epsilon_random_target

    def pick_action(self, state, episode, step, total_step):
        assert self._state_space is not None
        assert self._action_space is not None
        assert self._q_approximator is not None
        assert self._state_space.contains(state)

        if total_step < self._nb_rand_steps:
            self._this_step_rand_act = True
            return np.random.choice([0, 1, 2])
            #return self._action_space.sample()

        if self._force_random_action:
            self._force_random_action = False
            self._this_step_rand_act = True
            return np.random.choice([0, 1, 2])
            #return self._action_space.sample()

        if np.random.rand() < self._epsilon_random:
            # pick random action
            self._this_step_rand_act = True
            res = np.random.choice([0, 1, 2])
            #res = self._action_space.sample()

        else:
            self._this_step_rand_act = False
            # act greedy

            def rand_argmax(vector):
                """ Argmax that chooses randomly among eligible maximum indices. """
                vmax = np.max(vector)
                indices = np.nonzero(vector == vmax)[0]
                return np.random.choice(indices)

            obs = np.array([state])
            q_arr = self._q_approximator.estimate_all(obs).flatten()
            res = rand_argmax(q_arr)

        return res