import numpy as np
import matplotlib.pyplot as plt
import pdb

class MountainCarEnv:

    def __init__(self, log=None):
        self.t_step = 0
        self._pos = 0
        self._vel = 0
        self.reset()

        if log is not None:
            log.add_param('pos_min', -1.2)
            log.add_param('pos_max', 0.5)
            log.add_param('vel_min', -0.07)
            log.add_param('vel_max', 0.07)
            log.add_param('start_pos_min', -0.6)
            log.add_param('start_pos_max', -0.4)
            log.add_param('max_steps', float('+inf'))
            log.add_param('target_pos', 0.5)


    def reset(self, expl_start=False):
        self.t_step = 0
        if expl_start == True:
            self._pos = np.random.uniform(-1.2, 0.5)
            self._vel = np.random.uniform(-0.07, 0.07)
        else:
            self._pos = np.random.uniform(-0.6, -0.4)
            self._vel = 0.0
        self._done = False

        return np.array([self._pos, self._vel], dtype=float)


    def step(self, action):
        assert self._done == False
        assert action in [0, 1, 2]

        self.t_step += 1

        self._vel = self._vel + 0.001*(action-1) - 0.0025*np.cos(3*self._pos)
        self._vel = min(max(self._vel, -0.07), 0.07)

        self._pos = self._pos + self._vel
        self._pos = min(max(self._pos, -1.2), 0.5)

        if self._pos == -1.2:
            self._vel = 0.0

        if self._pos == 0.5:
            obs = np.array([self._pos, self._vel], dtype=float)
            reward = -1
            self._done = True
            return (obs, reward, self._done, None)
        else:
            obs = np.array([self._pos, self._vel], dtype=float)
            reward = -1
            return (obs, reward, self._done, None)


    def print_env(self):
        print('Env pos/vel: ', self._pos, self._vel)

