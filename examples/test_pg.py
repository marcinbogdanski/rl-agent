import numpy as np

import rl_agent as rl

import pdb


# env = rl.envs.Gridworld(4, 4, random_start=True)
# env.set_state(0, 3, 'terminal')
# env.set_state(3, 0, 'terminal')

env = rl.envs.Corridor(1, 0)
new_rewards = np.array([[-1, 1]])
assert env._rewards.shape == new_rewards.shape
env._rewards = new_rewards

# Under construction

agent = rl.AgentMonteCarlo(
    state_space=env.observation_space,
    action_space=env.action_space,
    discount=1.0,
    q_fun_approx=None,
    policy=rl.VanillaPolicyGradient(
        learn_rate=0.05)
    )

def _softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

done = True
while True:

    if done:
        print('=== episode ===')
        obs, rew, done = env.reset(), None, False
    else:
        print('===============')
        obs, rew, done = env.step(act)

    env.render()
    print('obs, rew, done:', obs, rew, done)

    agent.observe(obs, rew, done)

    # pdb.set_trace()
    print('weights', agent.policy._weights)
    print('prob', _softmax(agent.policy._weights[obs]))

    #pdb.set_trace()

    agent.learn()

    act = agent.take_action(obs)

    print('act', act)

    agent.next_step(done)









