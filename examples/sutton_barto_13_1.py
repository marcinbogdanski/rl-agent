import numpy as np
import matplotlib.pyplot as plt
import unittest

import rl_agent as rl
import gym

import pdb


class EnvSB_13_1:
    """Test envirnoment corresponding to Sutton and Barto 13.1 Nov 2017 draft.

    States are as follows:
     - 0 - left most state, start state, action left has no effect
     - 1 - middle state, has swapped left-right actions
     - 2 - right state, action right terminates episode
    All states look the same to the agent.

    Best policy is to move right with prob approx 0.59
    Best cumulative reward is approx -12
    """

    def __init__(self):
        self._state = 0
        self._is_done = False

        # 0 is move left, 1 is move right, swapped in central state
        self.action_space = gym.spaces.Discrete(2)
        
        # all states look the same, so we always return obs equal zero
        self.observation_space = gym.spaces.Discrete(1) 

    def reset(self):
        self._state = 0
        return 0  # observation is always zero

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError('Invalid action')
        if self._is_done:
            raise ValueError('Episode is terminated')
        
        if self._state == 0:
            # Left-most sate, NORMAL actions
            if action == 0:
                # do nothing
                return 0, -1, False
            else:
                self._state += 1
                return 0, -1, False
        elif self._state == 1:
            # Mid state, SWAPED actions
            if action == 0:
                self._state += 1  # go right
                return 0, -1, False
            else:
                self._state -= 1  # go left
                return 0, -1, False
        elif self._state == 2:
            # Righ-most state, NORMAL actions
            if action == 0:
                self._state -= 1  # go left
                return 0, -1, False
            else:
                self._state += 1  # go right
                return 0, -1, True
        else:
            raise ValueError('Invalid state')

    def render(self):
        print('current state:', self._state)            

def main(plot):
    np.random.seed(0)

    env = EnvSB_13_1()

    agent = rl.AgentOffline(
        state_space=env.observation_space,
        action_space=env.action_space,
        discount=1.0,
        q_fun_approx=None,
        policy=rl.VanillaPolicyGradient(
            learn_rate=0.00001)
        )


    if plot:
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        arr_tstep = []
        arr_pright = []
        arr_rew = []

    done = True
    i = -1
    while True:
        i += 1

        if done:

            if plot:
                arr_tstep.append(i)
                probs = agent.policy.get_raw(state=0)
                arr_pright.append(probs[1])

                if len(agent._episodes_history) > 100:
                    rews = [ep.total_reward for ep in agent._episodes_history[-100:]]
                    arr_rew.append(sum(rews) / len(rews))
                else:
                    arr_rew.append(0)

            if not plot and i >= 10000:
                print('{0[0]} {0[1]}'.format(agent.policy._weights[0]))
                return agent.policy._weights

            obs, rew, done = env.reset(), None, False
        else:
            obs, rew, done = env.step(act)

        agent.observe(obs, rew, done)

        agent.learn()

        act = agent.take_action(obs)

        agent.next_step(done)

        if plot and i % 1000 == 0:
            print('i', i)
            ax1.clear()
            ax1.plot(arr_tstep, arr_pright)
            ax2.clear()
            ax2.plot(arr_tstep, arr_rew)
            plt.pause(0.001)

if __name__ == '__main__':
    main(plot=False)


class TestMe(unittest.TestCase):
    def test_sutton_barto_13_1(self):
        result = main(plot=False)
        correct = np.array([[-0.02437331, 0.02437331]])
        self.assertTrue(result[0,0] == -0.02437330910810401)
        self.assertTrue(result[0,1] == 0.02437330910810403)







