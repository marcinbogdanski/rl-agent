


import matplotlib.pyplot as plt

from .agent import Agent

import pdb

def test_run(env, agent,

    nb_episodes, nb_total_steps, expl_start,

    plotter=None,
    seed=None):


    done = True

    while True:

        #   ---------------------------------
        #   ---   time step starts here   ---
        #   ---------------------------------

        

        if done:
            obs, reward, done = env.reset(), 0.0, False
        else:
            obs, reward, done, _ = env.step(action)

            reward = round(reward)


        agent.observe(obs, reward, done)

        agent.learn()
        
        if not done:
            action = agent.pick_action(obs)



        agent.log(agent.completed_episodes, agent.step, agent.total_step)

        if agent.total_step % 1000 == 0:
            print()
            print('total_step', agent.total_step,
                'e_rand', agent._epsilon_random,
                'step_size', agent._step_size)
            print('EPISODE', agent.completed_episodes)

        if plotter is not None and agent.total_step >= agent.nb_rand_steps:
            plotter.process(agent.logger, agent.total_step)
            res = plotter.conditional_plot(agent.logger, agent.total_step)
            if res:
                plt.pause(0.001)
                pass


        if done:
            print('espiode finished after iteration', agent.step)



        agent.advance_one_step(done)


        if  (nb_episodes is not None and agent.completed_episodes >= nb_episodes) or \
            (nb_total_steps is not None and agent.total_step > nb_total_steps):
                break



        

        #   ---------------------------------
        #   ---    time step ends here    ---
        #   ---------------------------------
        


    return agent