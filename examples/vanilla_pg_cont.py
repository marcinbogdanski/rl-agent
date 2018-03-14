import numpy as np
import matplotlib.pyplot as plt

import rl_agent as rl
import gym

import pdb

class EnvTarget:
    def __init__(self, target):
        self._low = 0
        self._high = 100
        self._target = target
        self._is_done = True

        self.action_space = gym.spaces.Box(
            low=float('-inf'), high=float('inf'), shape=(), dtype=np.float32)

        # we have only one state
        self.observation_space = gym.spaces.Discrete(1) 

    def reset(self):
        self._is_done = False
        return np.array(0)  # observation

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError('Invalid action')
        if self._is_done:
            raise ValueError('Episode is terminated')
        
        reward = -1 * abs(self._target - action)

        #  state, reward, done
        return np.array(0), reward, True

    def render(self):
        pass


env = EnvTarget(target=50)

# Under construction

agent = rl.AgentOffline(
    state_space=env.observation_space,
    action_space=env.action_space,
    discount=1.0,
    q_fun_approx=None,
    policy=rl.PolicyTabularCont(
        learn_rate=0.001, std_dev=1)
    )


fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
X = []
Y = []
R = []

#pdb.set_trace()

done = True
i = -1
while True:
    i += 1

    if done:

        X.append(i)
        means = agent.policy.get_raw(0)
        Y.append(means)

        # if len(agent._episodes_history) > 100:
        #     rews = [ep.total_reward for ep in agent._episodes_history[-100:]]
        #     R.append(sum(rews) / len(rews))
        # else:
        #     R.append(0)

        if i % 100 == 0:
            print('i', i)
            ax1.clear()
            ax1.plot(X, Y)
            # ax2.clear()
            # ax2.plot(X, R)
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
    #pdb.set_trace()
    agent.learn()

    act = agent.take_action(obs)

    # print('act', act)

    agent.next_step(done)









