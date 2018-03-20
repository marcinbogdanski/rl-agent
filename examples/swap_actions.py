import numpy as np
import matplotlib.pyplot as plt
import unittest

import rl_agent as rl
import gym

import pdb


def main(plot):
    np.random.seed(0)

    # Sutton Barto example 13.1
    env = rl.envs.SwapActions()

    agent = rl.AgentActorCritic(
        state_space=env.observation_space,
        action_space=env.action_space,
        discount=1.0,
        algorithm='mc_return',
        nb_episodes_in_batch=1,
        memory=rl.MemoryBasic(10000),
        v_fun_approx=None,
        q_fun_approx=None,
        policy=rl.PolicyTabularCat(
            learn_rate=0.00002)
        )


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

            # Plotting
            arr_tstep.append(i)
            probs = agent.policy.get_raw(state=0)
            arr_pright.append(probs[1])
            if len(agent._episodes_history) > 100:
                rews = [ep.total_reward for ep in agent._episodes_history[-100:]]
                arr_rew.append(sum(rews) / len(rews))
            else:
                arr_rew.append(0)

            obs, rew, done = env.reset(), None, False
        else:
            obs, rew, done = env.step(act)

        agent.observe(obs, rew, done)

        agent.learn()

        act = agent.take_action(obs)

        agent.next_step(done)

        if i % 1000 == 0:
            print('i', i)
            ax1.clear()
            ax1.plot(arr_tstep, arr_pright)
            ax2.clear()
            ax2.plot(arr_tstep, arr_rew)
            plt.pause(0.001)

if __name__ == '__main__':
    main(plot=True)









