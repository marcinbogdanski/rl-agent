


import matplotlib.pyplot as plt

from .agent import Agent

import pdb

g_plotter = None

def on_step_end(agent, reward, observation, done, action):
    global g_plotter

    if agent.total_step % 1000 == 0:
        print()
        print('total_step', agent.total_step,
            'e_rand', agent._epsilon_random,
            'step_size', agent._step_size)
        print('EPISODE', agent.completed_episodes)

    if g_plotter is not None and agent.total_step >= agent.nb_rand_steps:
        g_plotter.process(agent.logger, agent.total_step)
        res = g_plotter.conditional_plot(agent.logger, agent.total_step)
        if res:
            plt.pause(0.001)
            pass


    if done:
        print('espiode finished after iteration', agent.step)


def test_run(env, agent, nb_total_steps,

    plotter=None):

    global g_plotter  # yikes, hacky

    g_plotter = plotter


    done = True
    agent.reset()
    agent.register_callback('on_step_end', on_step_end)

    while True:

        #   ---------------------------------
        #   ---   time step starts here   ---
        #   ---------------------------------

        if done:
            obs, reward, done = env.reset(), None, False
        else:
            obs, reward, done, _ = env.step(action)


        agent.observe(obs, reward, done)

        agent.learn()
        
        action = agent.take_action(obs)

        agent.next_step(done)

        if agent.total_step > nb_total_steps:
            break     

        #   ---------------------------------
        #   ---    time step ends here    ---
        #   ---------------------------------
        

    agent.reset()
    agent.clear_callback('on_step_end')
    return agent