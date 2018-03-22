import numpy as np
import matplotlib.pyplot as plt

import rl_agent as rl
import gym

import pdb

class EnvFewAct:
    def __init__(self, rewards):
        self._rewards = np.array(rewards)
        self._is_done = True

        # only two actions
        self.action_space = gym.spaces.Discrete(3)

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
        
        reward = self._rewards[action]

        #  state, reward, done
        return np.array(0), reward, True

    def render(self):
        pass

np.random.seed(0)

#env = EnvFewAct(rewards=[-0.33, -0.33, 0.66])
env = EnvFewAct(rewards=[100, 100, 101])

# Under construction

agent = rl.AgentActorCritic(
    state_space=env.observation_space,
    action_space=env.action_space,
    discount=1.0,
    algorithm_type='mc_return',
    algorithm_mode='batch',
    algorithm_ep_in_batch=10,
    memory=rl.MemoryBasic(10000),
    v_fun_approx=None,
    q_fun_approx=None,
    policy=rl.PolicyTabularCat(learn_rate=0.001)
    )


fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
X = []
HL, HM, HR, PL, PM, PR = [], [], [], [], [], []
R = []

#pdb.set_trace()

done = True
i = -1
while True:
    i += 1

    print('  ---  ')

    if done:

        X.append(i)
        prob, weights = agent.policy.get_raw(0)
        print('weights', weights)
        print('prob', prob)
        PL.append(prob[0])
        PM.append(prob[1])
        PR.append(prob[2])
        HL.append(weights[0])
        HM.append(weights[1])
        HR.append(weights[2])

        # if len(agent._episodes_history) > 100:
        #     rews = [ep.total_reward for ep in agent._episodes_history[-100:]]
        #     R.append(sum(rews) / len(rews))
        # else:
        #     R.append(0)

        if i % 1 == 0:
            print('i', i)
            ax1.clear()
            ax2.clear()
            ax1.plot(X, PL, color='red', linestyle='-')
            ax1.plot(X, PM, color='blue', linestyle='-')
            ax1.plot(X, PR, color='green', linestyle='-')
            ax2.plot(X, HL, color='red', linestyle='-', alpha=0.3)
            ax2.plot(X, HM, color='blue', linestyle='-', alpha=0.3)
            ax2.plot(X, HR, color='green', linestyle='-', alpha=0.3)

            plt.pause(0.1)

        #pdb.set_trace()
        print('=== episode ===')
        obs, rew, done = env.reset(), None, False
    else:
        # print('===============')
        obs, rew, done = env.step(act)

    #env.render()
    # print('obs, rew, done:', obs, rew, done)

    agent.observe(obs, rew, done)

    # if done and i % 100 == 0:
    agent.learn()

    act = agent.take_action(obs)

    print('act', act)

    # print('act', act)

    agent.next_step(done)









