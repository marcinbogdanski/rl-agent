


import matplotlib.pyplot as plt

from .agent import Agent

import pdb

def test_run(env, agent,

    nb_episodes, nb_total_steps, expl_start,

    plotter=None,
    seed=None):

    #
    #   Loop episodes
    #
    episode = -1
    total_step = -1

    done = True

    while True:

        #   ---------------------------------
        #   ---   time step starts here   ---
        #   ---------------------------------

        if done:

            episode += 1           
            total_step += 1
            step = 0

            obs = env.reset()
            agent.reset()

            agent.append_trajectory(observation=obs,
                                    reward=None,
                                    done=None)

            done = False

        else:
            step += 1
            total_step += 1

            obs, reward, done, _ = env.step(action)

            reward = round(reward)

            agent.append_trajectory(
                        observation=obs,
                        reward=reward,
                        done=done)

            agent.eval_td_online()


        
        if not done:
            action = agent.pick_action(obs)
            agent.append_action(action=action)


        agent.log(episode, step, total_step)

        if total_step % 1000 == 0:
            print()
            print('total_step', total_step,
                'e_rand', agent._epsilon_random,
                'step_size', agent._step_size)

        if plotter is not None and total_step >= agent.nb_rand_steps:
            plotter.process(agent.logger, total_step)
            res = plotter.conditional_plot(agent.logger, total_step)
            if res:
                plt.pause(0.001)
                pass


        agent.advance_one_step()


        if  (nb_episodes is not None and episode >= nb_episodes) or \
            (nb_total_steps is not None and total_step >= nb_total_steps):
                break

        if done:
            print('espiode finished after iteration', step)

        #   ---------------------------------
        #   ---    time step ends here    ---
        #   ---------------------------------
        


    return agent