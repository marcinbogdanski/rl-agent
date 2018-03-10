import numpy as np
import matplotlib.pyplot as plt

import rl_agent as rl
import gym

import pdb

class EnvSB:
    def __init__(self):
        self._state = 0
        self._is_done = False

        self.action_space = gym.spaces.Discrete(2)
        # all states look the same
        self.observation_space = gym.spaces.Discrete(1) 

    def reset(self):
        self._state = 0
        return 0  # observation

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError('Invalid action')
        if self._is_done:
            raise ValueError('Episode is terminated')
        
        if self._state == 0:
            # NORMAL actions
            if action == 0:
                # do nothing
                return 0, -1, False
            else:
                self._state += 1
                return 0, -1, False
        elif self._state == 1:
            # SWAP actions
            if action == 0:
                self._state += 1  # go right
                return 0, -1, False
            else:
                self._state -= 1  # go left
                return 0, -1, False
        elif self._state == 2:
            # NORMAL actions
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



# env = rl.envs.Corridor(3, 1)
# new_rewards = np.array([[0, 0], [-1, 1], [0, 0]])
# assert env._rewards.shape == new_rewards.shape
# env._rewards = new_rewards

env = EnvSB()

# Under construction

agent = rl.AgentMonteCarlo(
    state_space=env.observation_space,
    action_space=env.action_space,
    discount=1.0,
    q_fun_approx=None,
    policy=rl.VanillaPolicyGradient(
        learn_rate=0.00001)
    )

def _softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)


fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
X = []
Y = []
R = []

done = True
i = -1
while True:
    i += 1

    if done:

        X.append(i)
        probs = _softmax(agent.policy._weights[0])
        Y.append(probs[1])

        if len(agent._episodes_history) > 100:
            rews = [ep.total_reward for ep in agent._episodes_history[-100:]]
            R.append(sum(rews) / len(rews))
        else:
            R.append(0)

        if i % 1000 == 0:
            print('i', i)
            ax1.clear()
            ax1.plot(X, Y)
            ax2.clear()
            ax2.plot(X, R)
            plt.pause(0.001)

        print('=== episode ===')
        obs, rew, done = env.reset(), None, False
    else:
        # print('===============')
        obs, rew, done = env.step(act)

    #env.render()
    # print('obs, rew, done:', obs, rew, done)

    agent.observe(obs, rew, done)

    # print('weights', agent.policy._weights)
    # print('prob', _softmax(agent.policy._weights[obs]))

    # if done and i % 100 == 0:
    #     pdb.set_trace()
    agent.learn()

    act = agent.take_action(obs)

    # print('act', act)

    agent.next_step(done)









