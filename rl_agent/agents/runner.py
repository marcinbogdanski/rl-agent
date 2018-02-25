


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

    initialise = True
       

    while True:

        if initialise:
            initialise = False


            episode += 1
            if nb_episodes is not None and episode >= nb_episodes:
                break

            if nb_total_steps is not None and total_step >= nb_total_steps:
                break


            step = 0
            total_step += 1


            obs = env.reset()
            agent.reset()

            agent.append_trajectory(observation=obs,
                                    reward=None,
                                    done=None)

            done = False





        
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

        if done or \
            (nb_total_steps is not None and total_step >= nb_total_steps):

            print('espiode finished after iteration', step)

            initialise = True

            continue

        #   --------------------------------
        #   ---   time step rolls here   ---
        #   --------------------------------

        step += 1
        total_step += 1

        obs, reward, done, _ = env.step(action)

        reward = round(reward)

        agent.append_trajectory(
                    observation=obs,
                    reward=reward,
                    done=done)

        agent.eval_td_online()
        


    return agent