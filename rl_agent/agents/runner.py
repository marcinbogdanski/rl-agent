


import matplotlib.pyplot as plt

from .agent import Agent

import pdb

def test_run(env, agent,

    nb_episodes, nb_total_steps, expl_start,

    agent_nb_actions,

    plotter=None,
    seed=None):

    #
    #   Loop episodes
    #
    episode = -1
    total_step = -1
    while True:

        PRINT_FROM = 100000
        
        episode += 1
        if nb_episodes is not None and episode >= nb_episodes:
            break

        if nb_total_steps is not None and total_step >= nb_total_steps:
            break

        
        step = 0
        total_step += 1

        print('episode:', episode, '/', nb_episodes,
            'step', step, 'total_step', total_step)


        print('-------------------------- total step -- ', total_step)

        obs = env.reset()
        print('STEP', obs)
        
        agent.reset()

        agent.append_trajectory(observation=obs,
                                reward=None,
                                done=None)

        while True:

            action = agent.pick_action(obs)
            if total_step >= PRINT_FROM:
                print('ACTION', action)
                print('MEM_LEN', agent._memory._curr_len)
                print('EPSILON', agent._epsilon_random)

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


            if nb_total_steps is not None and total_step >= nb_total_steps:
                break

            #   --------------------------------
            #   ---   time step rolls here   ---
            #   --------------------------------

            step += 1
            total_step += 1
            if total_step >= PRINT_FROM:
                print('------------------------------ total step -- ', total_step)


            if agent_nb_actions == 2 and action == 1:
                action_p = 2
            else:
                action_p = action
            obs, reward, done, _ = env.step(action_p)
            if total_step >= PRINT_FROM:
                print('STEP', obs)
            reward = round(reward)

            agent.append_trajectory(
                        observation=obs,
                        reward=reward,
                        done=done)

            agent.eval_td_online()
            
            if done or step >= 100000:
                print('espiode finished after iteration', step)
                
                agent.log(episode, step, total_step)

                if plotter is not None: # and total_step >= agent.nb_rand_steps:
                    plotter.process(agent.logger, total_step)

                agent.advance_one_step()
                break

    return agent